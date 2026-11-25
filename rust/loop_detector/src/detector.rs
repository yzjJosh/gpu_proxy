use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

const MAX_KGRAM_LEN: usize = 16;
const HASH_BASE: u64 = 0x9E37_79B9_7F4A_7C15;
const CODE_POINT_MIX: u64 = 0x517C_C1B7_2722_0A95;

/// Return a hash key unchanged.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0u64;
        for &byte in bytes {
            value = value.wrapping_shl(8) | (byte as u64);
        }
        self.0 = value;
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type AnchorTable = HashMap<u64, AnchorPositions, BuildHasherDefault<IdentityHasher>>;

enum AnchorPositions {
    Single(usize),
    Many(Vec<usize>),
}

impl AnchorPositions {
    #[inline]
    fn push(&mut self, pos: usize) {
        match self {
            AnchorPositions::Single(first) => {
                let first_pos = *first;
                *self = AnchorPositions::Many(vec![first_pos, pos]);
            }
            AnchorPositions::Many(positions) => positions.push(pos),
        }
    }

    /// Remove the latest position and return true if none remain.
    #[inline]
    fn pop_and_is_empty(&mut self) -> bool {
        match self {
            AnchorPositions::Single(_) => true,
            AnchorPositions::Many(positions) => {
                positions.pop();
                positions.is_empty()
            }
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        match self {
            AnchorPositions::Single(pos) => std::slice::from_ref(pos),
            AnchorPositions::Many(positions) => positions.as_slice(),
        }
    }
}

#[derive(Clone, Copy)]
struct WindowHashEntry {
    hash: u64,
    start_pos: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingCandidate {
    period: usize,
    target_end: usize,
}

#[inline]
fn mix_code_point(code_point: u32) -> u64 {
    (code_point as u64)
        .wrapping_add(1)
        .wrapping_mul(CODE_POINT_MIX)
}

#[inline]
fn finalize_hash(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Track a rolling hash over a fixed number of code points.
struct RollingHash {
    kgram_len: usize,
    oldest_code_point_weight: u64,
    raw_hash: u64,
    count: usize,
}

impl RollingHash {
    fn new(kgram_len: usize) -> Self {
        let oldest_code_point_weight = HASH_BASE.wrapping_pow((kgram_len - 1) as u32);
        Self {
            kgram_len,
            oldest_code_point_weight,
            raw_hash: 0,
            count: 0,
        }
    }

    fn reset(&mut self) {
        self.raw_hash = 0;
        self.count = 0;
    }

    /// Add one code point and return the window hash once enough code points are loaded.
    #[inline]
    fn push(&mut self, incoming: u32, outgoing: Option<u32>) -> Option<u64> {
        if self.count < self.kgram_len {
            self.raw_hash = self
                .raw_hash
                .wrapping_mul(HASH_BASE)
                .wrapping_add(mix_code_point(incoming));
            self.count += 1;
            if self.count == self.kgram_len {
                return Some(finalize_hash(self.raw_hash));
            }
            return None;
        }
        let outgoing_code_point =
            outgoing.expect("outgoing code point must exist once rolling hash is full");
        let without_oldest = self.raw_hash.wrapping_sub(
            mix_code_point(outgoing_code_point).wrapping_mul(self.oldest_code_point_weight),
        );
        self.raw_hash = without_oldest
            .wrapping_mul(HASH_BASE)
            .wrapping_add(mix_code_point(incoming));
        Some(finalize_hash(self.raw_hash))
    }
}

/// Error returned when feed is called after a loop without calling pop_last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedsRollbackError;

/// Detect repeated blocks in a stream of Unicode code points.
pub struct WinnowingLoopDetector {
    min_block_size: usize,
    kgram_len: usize,
    window_size: usize,
    buffer: Vec<u32>,
    rolling_hash: RollingHash,
    monotonic_deque: VecDeque<WindowHashEntry>,
    last_emitted_minimizer: Option<usize>,
    anchor_table: AnchorTable,
    pending_candidates: Vec<PendingCandidate>,
    rollback_buffer_len: usize,
    can_rollback: bool,
    needs_rollback_after_loop: bool,
    latest_feed_anchor_hashes: Vec<u64>,
}

impl WinnowingLoopDetector {
    /// Create an empty detector for repeats of at least min_block_size code points.
    pub fn new(min_block_size: usize) -> Self {
        assert!(min_block_size >= 1, "min_block_size must be at least 1");
        let kgram_len = (min_block_size / 2).clamp(1, MAX_KGRAM_LEN);
        let window_size = min_block_size - kgram_len + 1;
        Self {
            min_block_size,
            kgram_len,
            window_size,
            buffer: Vec::new(),
            rolling_hash: RollingHash::new(kgram_len),
            monotonic_deque: VecDeque::with_capacity(window_size + 1),
            last_emitted_minimizer: None,
            anchor_table: AnchorTable::default(),
            pending_candidates: Vec::new(),
            rollback_buffer_len: 0,
            can_rollback: false,
            needs_rollback_after_loop: false,
            latest_feed_anchor_hashes: Vec::new(),
        }
    }

    /// Create a detector holding the trailing part of code_points after any prior repeat.
    pub fn from_code_points(code_points: Vec<u32>, min_block_size: usize) -> Self {
        assert!(min_block_size >= 1, "min_block_size must be at least 1");
        let len = code_points.len();
        let mut detector = Self::new(min_block_size);
        detector.buffer = code_points;
        let mut has_loop = false;
        for scanned_len in 1..=len {
            if detector.scan_code_point_at(scanned_len, true, false).is_some() {
                has_loop = true;
                break;
            }
        }
        if !has_loop {
            return detector;
        }
        let mut code_points = detector.buffer;
        let safe_start = Self::find_safe_suffix_start(&code_points, min_block_size);
        if safe_start > 0 {
            code_points.drain(..safe_start);
        }
        let mut clean_detector = Self::new(min_block_size);
        clean_detector.buffer = code_points;
        let clean_len = clean_detector.buffer.len();
        for scanned_len in 1..=clean_len {
            clean_detector.scan_code_point_at(scanned_len, false, false);
        }
        if clean_len >= min_block_size {
            clean_detector.sync_pending_from_deque_front();
        }
        clean_detector
    }

    /// Return true if the detector is waiting for pop_last after a repeat.
    #[cfg(test)]
    #[inline]
    pub fn needs_rollback(&self) -> bool {
        self.needs_rollback_after_loop
    }

    /// Return the current buffer length before a feed, or an error if a prior loop was not rolled back.
    pub fn begin_feed(&mut self) -> Result<usize, NeedsRollbackError> {
        if self.needs_rollback_after_loop {
            return Err(NeedsRollbackError);
        }
        Ok(self.buffer.len())
    }

    /// Return mutable access to the code point buffer.
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut Vec<u32> {
        &mut self.buffer
    }

    /// Check code points appended after old_len and return true if a repeat completes.
    pub fn finish_feed(&mut self, old_len: usize) -> bool {
        let new_len = self.buffer.len();
        if new_len == old_len {
            return false;
        }
        self.rollback_buffer_len = old_len;
        self.can_rollback = true;
        self.latest_feed_anchor_hashes.clear();
        for scanned_len in (old_len + 1)..=new_len {
            if self.scan_code_point_at(scanned_len, true, true).is_some() {
                self.needs_rollback_after_loop = true;
                return true;
            }
        }
        false
    }

    /// Append delta and return true if a repeat completes.
    #[cfg(test)]
    pub fn feed_slice(&mut self, delta: &[u32]) -> Result<bool, NeedsRollbackError> {
        let old_len = self.begin_feed()?;
        if delta.is_empty() {
            return Ok(false);
        }
        self.buffer.extend_from_slice(delta);
        Ok(self.finish_feed(old_len))
    }

    /// Undo the most recent non-empty feed call and return true if one was undone.
    pub fn pop_last(&mut self) -> bool {
        if !self.can_rollback {
            return false;
        }
        for hash in self.latest_feed_anchor_hashes.drain(..).rev() {
            let remove_bucket = match self.anchor_table.get_mut(&hash) {
                Some(bucket) => bucket.pop_and_is_empty(),
                None => false,
            };
            if remove_bucket {
                self.anchor_table.remove(&hash);
            }
        }
        let target_len = self.rollback_buffer_len;
        self.buffer.truncate(target_len);
        self.resync_tail_state();
        self.can_rollback = false;
        self.needs_rollback_after_loop = false;
        true
    }

    /// Return the start index after dropping the first copy of any trailing repeat.
    fn find_safe_suffix_start(code_points: &[u32], min_block_size: usize) -> usize {
        let total = code_points.len();
        if total < 2 * min_block_size {
            return 0;
        }
        let mut reverse_detector = Self::new(min_block_size);
        reverse_detector.buffer.reserve(total);
        for (index, &code_point) in code_points.iter().rev().enumerate() {
            reverse_detector.buffer.push(code_point);
            let scanned_len = index + 1;
            if let Some(period) = reverse_detector.scan_code_point_at(scanned_len, true, false) {
                // Skip the first copy of the trailing repeat and keep its second copy.
                return total - scanned_len + period;
            }
        }
        0
    }

    /// Rebuild the trailing window state from the current buffer.
    fn resync_tail_state(&mut self) {
        self.rolling_hash.reset();
        self.monotonic_deque.clear();
        self.pending_candidates.clear();
        self.last_emitted_minimizer = None;

        let len = self.buffer.len();
        let tail_start = len.saturating_sub(self.min_block_size);
        for index in tail_start..len {
            let scanned_len = index + 1;
            let outgoing = if scanned_len > tail_start + self.kgram_len {
                Some(self.buffer[index - self.kgram_len])
            } else {
                None
            };
            if let Some(hash) = self.rolling_hash.push(self.buffer[index], outgoing) {
                let kgram_start = scanned_len - self.kgram_len;
                self.push_window_hash(hash, kgram_start);
            }
        }
        if len >= self.min_block_size {
            if let Some(&front_entry) = self.monotonic_deque.front() {
                self.last_emitted_minimizer = Some(front_entry.start_pos);
                self.sync_pending_from_deque_front();
            }
        }
    }

    /// Refresh pending repeat candidates at the end of the buffer.
    fn sync_pending_from_deque_front(&mut self) {
        self.pending_candidates.clear();
        let Some(&min_entry) = self.monotonic_deque.front() else {
            return;
        };
        let scanned_len = self.buffer.len();
        self.check_candidates_for_minimizer(
            min_entry.hash,
            min_entry.start_pos,
            scanned_len,
            false,
        );
    }

    /// Scan one code point and return the shortest repeat length if a repeat completes.
    fn scan_code_point_at(
        &mut self,
        scanned_len: usize,
        check_loops: bool,
        track_rollback: bool,
    ) -> Option<usize> {
        let pending_period = check_loops
            .then(|| self.check_pending_candidates(scanned_len))
            .flatten();
        let index = scanned_len - 1;
        let outgoing = (scanned_len > self.kgram_len).then(|| self.buffer[index - self.kgram_len]);
        let Some(hash) = self.rolling_hash.push(self.buffer[index], outgoing) else {
            return pending_period;
        };
        self.push_window_hash(hash, scanned_len - self.kgram_len);
        if scanned_len < self.min_block_size {
            return pending_period;
        }
        let &min_entry = self
            .monotonic_deque
            .front()
            .expect("deque cannot be empty once scanned_len >= min_block_size");
        if self.last_emitted_minimizer == Some(min_entry.start_pos) {
            return pending_period;
        }
        let minimizer_period = check_loops
            .then(|| {
                self.check_candidates_for_minimizer(
                    min_entry.hash,
                    min_entry.start_pos,
                    scanned_len,
                    true,
                )
            })
            .flatten();
        if let Some(period) = [pending_period, minimizer_period]
            .into_iter()
            .flatten()
            .min()
        {
            return Some(period);
        }
        self.record_anchor(min_entry.hash, min_entry.start_pos, track_rollback);
        self.last_emitted_minimizer = Some(min_entry.start_pos);
        None
    }

    /// Check pending candidates against the latest code point and return the shortest completed repeat length.
    fn check_pending_candidates(&mut self, scanned_len: usize) -> Option<usize> {
        if self.pending_candidates.is_empty() {
            return None;
        }
        let last_index = scanned_len - 1;
        let last_code_point = self.buffer[last_index];
        let mut write_index = 0;
        let mut completed_period: Option<usize> = None;
        for read_index in 0..self.pending_candidates.len() {
            let candidate = self.pending_candidates[read_index];
            if self.buffer[last_index - candidate.period] == last_code_point {
                if scanned_len == candidate.target_end {
                    completed_period = Some(match completed_period {
                        Some(existing) => existing.min(candidate.period),
                        None => candidate.period,
                    });
                }
                self.pending_candidates[write_index] = candidate;
                write_index += 1;
            }
        }
        self.pending_candidates.truncate(write_index);
        completed_period
    }

    #[inline]
    fn push_window_hash(&mut self, hash: u64, kgram_start: usize) {
        while let Some(&back_entry) = self.monotonic_deque.back() {
            if back_entry.hash >= hash {
                self.monotonic_deque.pop_back();
            } else {
                break;
            }
        }
        self.monotonic_deque.push_back(WindowHashEntry {
            hash,
            start_pos: kgram_start,
        });
        while let Some(&front_entry) = self.monotonic_deque.front() {
            if front_entry.start_pos + self.window_size <= kgram_start {
                self.monotonic_deque.pop_front();
            } else {
                break;
            }
        }
    }

    /// Check prior positions for min_hash and return the shortest repeat length if found.
    fn check_candidates_for_minimizer(
        &mut self,
        min_hash: u64,
        min_pos: usize,
        scanned_len: usize,
        report_complete_repeat: bool,
    ) -> Option<usize> {
        let bucket = self.anchor_table.get(&min_hash)?;
        for &prior_pos in bucket.as_slice().iter().rev() {
            if prior_pos >= min_pos {
                continue;
            }
            let period = min_pos - prior_pos;
            if 2 * period > min_pos + self.min_block_size {
                break;
            }
            if period < self.min_block_size
                || !self.passes_repeat_prefilter(prior_pos, min_pos, period, scanned_len)
            {
                continue;
            }
            let forward_limit = period.min(scanned_len - min_pos);
            let forward_match_len = self.count_matching_forward(prior_pos, min_pos, forward_limit);
            let reaches_end = forward_match_len == scanned_len - min_pos;
            if forward_match_len < self.kgram_len || (!report_complete_repeat && !reaches_end) {
                continue;
            }
            let backward_match_len =
                self.count_matching_backward(prior_pos, min_pos, period.min(prior_pos));
            if backward_match_len + forward_match_len >= period {
                if report_complete_repeat {
                    return Some(period);
                }
                continue;
            }
            if reaches_end {
                let target_end = min_pos - backward_match_len + period;
                if target_end > scanned_len && target_end <= min_pos + self.min_block_size {
                    register_pending_candidate(&mut self.pending_candidates, period, target_end);
                }
            }
        }
        None
    }

    /// Return false if two anchor positions cannot form a repeat.
    #[inline]
    fn passes_repeat_prefilter(
        &self,
        prior_pos: usize,
        current_pos: usize,
        period: usize,
        scanned_len: usize,
    ) -> bool {
        // Anchor endpoints must match.
        if self.buffer[prior_pos] != self.buffer[current_pos]
            || self.buffer[prior_pos + self.kgram_len - 1]
                != self.buffer[current_pos + self.kgram_len - 1]
        {
            return false;
        }
        // Matches wider than min_block_size must extend to the left.
        let required_left = period - self.min_block_size;
        if required_left >= 1
            && (self.buffer[prior_pos - 1] != self.buffer[current_pos - 1]
                || self.buffer[prior_pos - required_left]
                    != self.buffer[current_pos - required_left])
        {
            return false;
        }
        // Interior anchors must match on at least one side.
        if current_pos + self.kgram_len < scanned_len {
            let left_probe_offset = period / 2;
            let right_probe_offset = period.div_ceil(2).min(scanned_len - 1 - current_pos);
            let left_ok = prior_pos >= left_probe_offset
                && self.buffer[prior_pos - left_probe_offset]
                    == self.buffer[current_pos - left_probe_offset];
            let right_ok = self.buffer[prior_pos + right_probe_offset]
                == self.buffer[current_pos + right_probe_offset];
            if !left_ok && !right_ok {
                return false;
            }
        }
        true
    }

    #[inline]
    fn count_matching_forward(
        &self,
        prior_pos: usize,
        current_pos: usize,
        max_len: usize,
    ) -> usize {
        let prior_slice = &self.buffer[prior_pos..prior_pos + max_len];
        let current_slice = &self.buffer[current_pos..current_pos + max_len];
        prior_slice
            .iter()
            .zip(current_slice.iter())
            .take_while(|(left, right)| left == right)
            .count()
    }

    #[inline]
    fn count_matching_backward(
        &self,
        prior_pos: usize,
        current_pos: usize,
        max_len: usize,
    ) -> usize {
        let prior_slice = &self.buffer[prior_pos - max_len..prior_pos];
        let current_slice = &self.buffer[current_pos - max_len..current_pos];
        prior_slice
            .iter()
            .rev()
            .zip(current_slice.iter().rev())
            .take_while(|(left, right)| left == right)
            .count()
    }

    #[inline]
    fn record_anchor(&mut self, hash: u64, pos: usize, track_rollback: bool) {
        self.anchor_table
            .entry(hash)
            .and_modify(|bucket| bucket.push(pos))
            .or_insert(AnchorPositions::Single(pos));
        if track_rollback {
            self.latest_feed_anchor_hashes.push(hash);
        }
    }
}

#[inline]
fn register_pending_candidate(
    pending: &mut Vec<PendingCandidate>,
    period: usize,
    target_end: usize,
) {
    for existing in pending.iter_mut() {
        if existing.period == period {
            if target_end < existing.target_end {
                existing.target_end = target_end;
            }
            return;
        }
    }
    pending.push(PendingCandidate { period, target_end });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Return true if buffer contains a contiguous repeat of at least min_block_size code points.
    fn brute_force_has_square(buffer: &[u32], min_block_size: usize) -> bool {
        let len = buffer.len();
        for period in min_block_size..=(len / 2) {
            for start in 0..=(len - 2 * period) {
                if buffer[start..start + period] == buffer[start + period..start + 2 * period] {
                    return true;
                }
            }
        }
        false
    }

    /// Return the shortest prefix length of buffer that ends with a repeat.
    fn brute_force_first_square_end(buffer: &[u32], min_block_size: usize) -> Option<usize> {
        for end in (2 * min_block_size)..=buffer.len() {
            for period in min_block_size..=(end / 2) {
                if buffer[end - 2 * period..end - period] == buffer[end - period..end] {
                    return Some(end);
                }
            }
        }
        None
    }

    /// Return the suffix start after dropping the first copy of the last repeat.
    fn brute_force_safe_suffix_start(buffer: &[u32], min_block_size: usize) -> usize {
        let len = buffer.len();
        for start in (0..len).rev() {
            let max_period = (len - start) / 2;
            for period in min_block_size..=max_period {
                if buffer[start..start + period] == buffer[start + period..start + 2 * period] {
                    return start + period;
                }
            }
        }
        0
    }

    fn encode_code_points(text: &str) -> Vec<u32> {
        text.chars().map(|character| character as u32).collect()
    }

    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            state
        }

        fn next_range(&mut self, upper: usize) -> usize {
            (self.next() as usize) % upper
        }
    }

    #[test]
    fn exact_boundary_repeated_single_char_50() {
        let mut detector = WinnowingLoopDetector::new(50);
        let block = encode_code_points(&"a".repeat(50));
        assert_eq!(detector.feed_slice(&block), Ok(false));
        assert_eq!(
            detector.feed_slice(&encode_code_points(&"a".repeat(49))),
            Ok(false)
        );
        assert_eq!(detector.feed_slice(&encode_code_points("a")), Ok(true));
        assert!(detector.needs_rollback());
        assert_eq!(
            detector.feed_slice(&encode_code_points("b")),
            Err(NeedsRollbackError)
        );
        assert_eq!(detector.feed_slice(&[]), Err(NeedsRollbackError));
        assert!(detector.pop_last());
        assert!(!detector.needs_rollback());
        assert_eq!(detector.buffer.len(), 99);
        assert!(!detector.pop_last());
        assert_eq!(detector.feed_slice(&encode_code_points("b")), Ok(false));
        assert_eq!(detector.buffer.len(), 100);
    }

    #[test]
    fn rightward_mismatch_does_not_trigger_false_repeat() {
        // A matching prefix with a later mismatch is not a repeat.
        let buffer = encode_code_points("zzabcdzzabXd");
        assert!(!brute_force_has_square(&buffer, 4));
        let mut detector = WinnowingLoopDetector::new(4);
        assert_eq!(detector.feed_slice(&buffer), Ok(false));
    }

    #[test]
    fn rollback_into_partial_window_below_min_block_size() {
        let mut detector = WinnowingLoopDetector::new(50);
        let prefix = encode_code_points("0123456789abcdefghij");
        assert_eq!(detector.feed_slice(&prefix), Ok(false));
        assert_eq!(
            detector.feed_slice(&encode_code_points("ZZZZZZZZZZ")),
            Ok(false)
        );
        assert!(detector.pop_last());
        assert_eq!(detector.buffer.len(), 20);
        let rest_of_block = encode_code_points("klmnopqrstuvwxyz0123456789ABCD");
        assert_eq!(detector.feed_slice(&rest_of_block), Ok(false));
        let mut full_block = prefix.clone();
        full_block.extend_from_slice(&rest_of_block);
        assert_eq!(detector.feed_slice(&full_block[..49]), Ok(false));
        assert_eq!(detector.feed_slice(&full_block[49..50]), Ok(true));
    }

    #[test]
    fn unicode_code_point_length_semantics() {
        // Fifty arrows take 150 UTF-8 bytes and 50 code points.
        let mut detector = WinnowingLoopDetector::new(50);
        let block = encode_code_points(&"→".repeat(50));
        assert_eq!(detector.feed_slice(&block), Ok(false));
        assert_eq!(
            detector.feed_slice(&encode_code_points(&"→".repeat(49))),
            Ok(false)
        );
        assert_eq!(detector.feed_slice(&encode_code_points("→")), Ok(true));
        assert!(detector.pop_last());
        assert_eq!(detector.buffer.len(), 99);
    }

    #[test]
    fn differential_streaming_0_lag_against_oracle() {
        let mut rng = XorShift64(0xDEAD_BEEF_CAFE_BABE);
        for &min_block_size in &[1, 2, 3, 5, 16, 32, 50] {
            for trial in 0..80 {
                let alphabet = if trial % 3 == 0 {
                    2
                } else if trial % 3 == 1 {
                    4
                } else {
                    12
                };
                let len = min_block_size * 4 + rng.next_range(min_block_size * 2 + 10);
                let mut text: Vec<u32> = (0..len)
                    .map(|_| ('a' as u32) + (rng.next_range(alphabet) as u32))
                    .collect();
                // Add a repeat to half of the trials.
                if trial % 2 == 0 {
                    let period = min_block_size + rng.next_range(min_block_size + 5);
                    if 2 * period <= text.len() {
                        let start = rng.next_range(text.len() - 2 * period + 1);
                        let copy: Vec<u32> = text[start..start + period].to_vec();
                        text[start + period..start + 2 * period].copy_from_slice(&copy);
                    }
                }
                let expected_end = brute_force_first_square_end(&text, min_block_size);

                let mut char_detector = WinnowingLoopDetector::new(min_block_size);
                let mut actual_end = None;
                for (index, &code_point) in text.iter().enumerate() {
                    if char_detector.feed_slice(&[code_point]).unwrap() {
                        actual_end = Some(index + 1);
                        break;
                    }
                }
                assert_eq!(
                    actual_end, expected_end,
                    "1-char 0-lag mismatch for min_block_size={min_block_size}, trial={trial}"
                );

                let mut chunk_detector = WinnowingLoopDetector::new(min_block_size);
                let mut pos = 0;
                let mut detected_chunk = false;
                while pos < text.len() {
                    let step = 1 + rng.next_range(min_block_size + 7);
                    let next_pos = (pos + step).min(text.len());
                    if chunk_detector.feed_slice(&text[pos..next_pos]).unwrap() {
                        detected_chunk = true;
                        assert!(
                            expected_end.is_some() && expected_end.unwrap() <= next_pos,
                            "chunk false positive for min_block_size={min_block_size}, trial={trial}"
                        );
                        break;
                    }
                    pos = next_pos;
                }
                assert_eq!(
                    detected_chunk,
                    expected_end.is_some(),
                    "chunk detection mismatch for min_block_size={min_block_size}, trial={trial}"
                );
            }
        }
    }

    #[test]
    fn differential_from_history_and_pop_last_against_oracle() {
        let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);
        for &min_block_size in &[1, 2, 5, 16, 35, 50] {
            for trial in 0..60 {
                let alphabet = if trial % 2 == 0 { 2 } else { 5 };
                let history_len = rng.next_range(min_block_size * 4 + 10);
                let mut history: Vec<u32> = (0..history_len)
                    .map(|_| ('a' as u32) + (rng.next_range(alphabet) as u32))
                    .collect();
                if history_len >= 2 * min_block_size && trial % 3 == 0 {
                    let period =
                        min_block_size + rng.next_range((history_len / 2 - min_block_size) + 1);
                    let start = rng.next_range(history_len - 2 * period + 1);
                    let copy: Vec<u32> = history[start..start + period].to_vec();
                    history[start + period..start + 2 * period].copy_from_slice(&copy);
                }

                let expected_safe_start = brute_force_safe_suffix_start(&history, min_block_size);
                let mut detector =
                    WinnowingLoopDetector::from_code_points(history.clone(), min_block_size);
                assert_eq!(
                    detector.buffer,
                    &history[expected_safe_start..],
                    "safe suffix mismatch for min_block_size={min_block_size}, trial={trial}"
                );
                assert!(!detector.pop_last());

                let mut oracle_buffer = history[expected_safe_start..].to_vec();
                for _step in 0..12 {
                    let delta_len = rng.next_range(min_block_size + 5);
                    let delta: Vec<u32> = (0..delta_len)
                        .map(|_| ('a' as u32) + (rng.next_range(alphabet) as u32))
                        .collect();
                    let mut candidate_buffer = oracle_buffer.clone();
                    candidate_buffer.extend_from_slice(&delta);
                    let expected_loop = brute_force_has_square(&candidate_buffer, min_block_size);
                    let got_loop = detector.feed_slice(&delta).unwrap();
                    assert_eq!(
                        got_loop, expected_loop,
                        "post-history feed mismatch for min_block_size={min_block_size}, trial={trial}"
                    );
                    if got_loop {
                        assert!(detector.pop_last());
                        assert_eq!(detector.buffer, oracle_buffer);
                    } else if !delta.is_empty() && rng.next_range(3) == 0 {
                        assert!(detector.pop_last());
                        assert_eq!(detector.buffer, oracle_buffer);
                    } else {
                        oracle_buffer = candidate_buffer;
                    }
                }
            }
        }
    }

    #[test]
    fn boilerplate_newly_entering_and_interior_minimizers() {
        // Repeated boilerplate separated by unique markers does not trigger a loop.
        let mut text = String::new();
        for turn in 0..200 {
            text.push_str(&format!(
                "1. `deadbeef-0000-4000-8000-000000000001` - turn {turn:04} details\n"
            ));
        }
        let code_points = encode_code_points(&text);
        let mut detector = WinnowingLoopDetector::from_code_points(code_points.clone(), 150);
        assert_eq!(detector.buffer.len(), code_points.len());
        let tail = code_points[code_points.len() - 150..].to_vec();
        assert_eq!(detector.feed_slice(&tail[..149]), Ok(false));
        assert_eq!(detector.feed_slice(&tail[149..150]), Ok(true));
    }

    #[test]
    fn from_code_points_picks_shortest_period_on_simultaneous_repeats() {
        let min_block_size = 4;
        for sample in [
            "j+@jj+@j+@jj+@8@GO",
            ",KX,,KX,KX,,KX7^",
            "mEammEamEammEaQ+(5",
        ] {
            let history = encode_code_points(sample);
            let expected_safe_start = brute_force_safe_suffix_start(&history, min_block_size);
            let mut detector =
                WinnowingLoopDetector::from_code_points(history.clone(), min_block_size);
            assert_eq!(
                detector.buffer,
                &history[expected_safe_start..],
                "simultaneous repeat safe suffix mismatch for sample={sample}"
            );
            assert_eq!(
                detector.feed_slice(&history[expected_safe_start..]),
                Ok(true),
                "re-feeding retained suffix should trigger repeat for sample={sample}"
            );
        }
    }
}

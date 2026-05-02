use std::collections::VecDeque;
use std::convert::Infallible;

pub struct MockRng {
    bytes: VecDeque<u64>,
}
impl MockRng {
    pub fn new() -> Self {
        Self {bytes: VecDeque::new()}
    }

    pub fn push_u64(&mut self, n: u64) {
        self.bytes.push_back(n);
    }

    pub fn push_bool(&mut self, b: bool) {
        if b {
            self.bytes.push_back(u64::MIN);
        }
        else {
            self.bytes.push_back(u64::MAX);
        }
    }

    pub fn push_index(&mut self, target: usize, range: usize) {
        if range <= u32::MAX as usize {
            let r = range as u32;
            let x = ((((target as u128) << 32) + (range as u128) - 1)
                / (range as u128)) as u32;

            self.bytes.push_back(x as u64);

            let lo_order = ((x as u64) * (r as u64)) as u32;
            if lo_order > r.wrapping_neg() {
                self.bytes.push_back(0);
            }
        } else {
            let r = range as u64;

            let x = ((((target as u128) << 64) + (range as u128) - 1)
                / (range as u128)) as u64;

            self.bytes.push_back(x);

            let lo_order = ((x as u128) * (r as u128)) as u64;
            if lo_order > r.wrapping_neg() {
                self.bytes.push_back(0);
            }
        }
    }

    pub fn push_range_float(&mut self, expected_value: f64, lower: f64, upper: f64) -> f64 {
        let fraction: f64 = (expected_value - lower) / (upper - lower);
        let fraction = fraction.clamp(0.0, 1.0);

        let max_u52: u64 = (1u64 << 52) - 1;
        let u52 = (fraction * max_u52 as f64).round() as u64;

        let raw_u64 = u52 << 12;
        self.bytes.push_back(raw_u64);

        let actual_fraction = u52 as f64 / (1u64 << 52) as f64;
        let actual_value = actual_fraction * (upper - lower) + lower;
        actual_value
    }
    pub fn push_uniform_float_inclusive(&mut self, expected_value: f64, lower: f64, upper: f64) -> f64 {
        if lower == upper {
            self.bytes.push_back(0);
            return lower;
        }

        const F64_DENOM: u64 = 1u64 << 52;
        const MAX_U52: u64 = F64_DENOM - 1;

        let max_rand = MAX_U52 as f64 / F64_DENOM as f64;
        let mut scale: f64 = (upper - lower) / max_rand;

        fn next_down_positive_f64(x: f64) -> f64 {
            f64::from_bits(x.to_bits() - 1)
        }

        while scale * max_rand + lower > upper {
            scale = next_down_positive_f64(scale);
        }

        let fraction = ((expected_value - lower) / scale).clamp(0.0, max_rand);
        let u52 = ((fraction * F64_DENOM as f64).round() as u64).min(MAX_U52);

        self.bytes.push_back(u52 << 12);

        let actual_fraction = u52 as f64 / F64_DENOM as f64;
        actual_fraction * scale + lower
    }
}
impl rand_core::TryRng for MockRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64().unwrap() as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> { {
        let val = self.bytes.pop_front().expect(
            "MockRng should not run out of bytes."
        );
        Ok(val)
    }}

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let mut i = 0;
        while i < dst.len() {
            let random_bytes: [u8; 8] = self.try_next_u64().unwrap().to_le_bytes();
            let copy_len = std::cmp::min(8, dst.len() - i);
            dst[i..i + copy_len].copy_from_slice(&random_bytes[..copy_len]);
            i += copy_len;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_mock_rng {
    use rand::distr::Uniform;
    use super::*;
    use test_case::test_case;
    use rand_core::Rng;
    use rand::prelude::*;

    #[test_case(0xF5963A44 ; "0xF5963A44")]
    #[test_case(0xA59AFCEE ; "0xA59AFCEE")]
    #[test_case(0xD2FCB41C ; "0xD2FCB41C")]
    #[test_case(0x0A46DC65 ; "0x0A46DC65")]
    #[test_case(0xDB663CE1 ; "0xDB663CE1")]
    #[test_case(0xA1FD63CF ; "0xA1FD63CF")]
    #[test_case(0x51EB2E90 ; "0x51EB2E90")]
    #[test_case(0x192D2110 ; "0x192D2110")]
    #[test_case(0x90DF3816 ; "0x90DF3816")]
    #[test_case(0x80F0C19C ; "0x80F0C19C")]
    fn test_push_u64(n: u64) {
        let mut mock_rng = MockRng::new();
        mock_rng.push_u64(n);
        assert_eq!(mock_rng.next_u64(), n)
    }

    #[test]
    fn test_repeated_u64() {
        let mut mock_rng = MockRng::new();
        let u64_vec: Vec<u64> = vec![0xF5963A44, 0xA59AFCEE, 0xD2FCB41C, 0x0A46DC65, 0xDB663CE1,
                                     0xA1FD63CF, 0x51EB2E90, 0x192D2110, 0x90DF3816, 0x80F0C19C];

        for &n in u64_vec.iter() {
            mock_rng.push_u64(n);
        }

        for &n in u64_vec.iter() {
            assert_eq!(mock_rng.next_u64(), n)
        }
    }

    #[test]
    fn test_push_bool() {
        let mut mock_rng = MockRng::new();
        mock_rng.push_bool(true);
        assert!(mock_rng.random_bool(0.0001));
        mock_rng.push_bool(false);
        assert!(!mock_rng.random_bool(0.9999));
    }

    #[test]
    fn test_repeated_push_bool() {
        let mut mock_rng = MockRng::new();
        let bool_vec: Vec<bool> = vec![true, false, false, false, true, true, true, false, true, true];

        for &b in bool_vec.iter() {
            mock_rng.push_bool(b);
        }

        for &b in bool_vec.iter() {
            if b {
                assert!(mock_rng.random_bool(0.0001));
            }
            else {
                assert!(!mock_rng.random_bool(0.9999));
            }
        }
    }

    #[test_case( 0, 78;  "0 78")]
    #[test_case(85, 86; "85 86")]
    #[test_case(36, 66; "36 66")]
    #[test_case(21, 55; "21 55")]
    #[test_case(76, 96; "76 96")]
    #[test_case( 2,  5;   "2 5")]
    #[test_case(14647823, 63528435; "14647823 63528435")]
    fn test_push_index(t: usize, r: usize) {
        let mut mock_rng = MockRng::new();
        mock_rng.push_index(t, r);
        assert_eq!(mock_rng.random_range(0..r), t)
    }

    #[test]
    fn test_repeated_push_index() {
        let mut mock_rng = MockRng::new();
        let range_vec: Vec<(usize, usize)> = vec![(0, 76), (85, 86), (36, 66), (21, 55), (76, 96),
                                                  (2, 5), (14647823, 63528435)];

        for &(t, r) in range_vec.iter() {
            mock_rng.push_index(t, r);
        }

        for &(t, r) in range_vec.iter() {
           assert_eq!(mock_rng.random_range(0..r), t)
        }
    }

    #[test_case(0.5, 0.0, 1.0 ; "standard mid range")]
    #[test_case(0.0, 0.0, 1.0 ; "exact lower bound")]
    #[test_case(0.999999, 0.0, 1.0 ; "close to upper bound")]
    #[test_case(-3.14, -10.0, -2.0 ; "negative range")]
    #[test_case(0.0, -5.5, 5.5 ; "perfect zero")]
    #[test_case(42.4242, 40.0, 50.0 ; "arbitrary positive floats")]
    #[test_case(0.00015, 0.0001, 0.0002 ; "microscopic range")]
    #[test_case(1_000_000.5, 0.0, 2_000_000.0 ; "massive bounds")]
    fn test_push_range_float(expected: f64, lower: f64, upper: f64) {
        let mut mock_rng = MockRng::new();

        let actual_mocked_val = mock_rng.push_range_float(expected, lower, upper);
        assert_eq!(mock_rng.random_range(lower..upper), actual_mocked_val);
        let actual_mocked_val = mock_rng.push_range_float(expected, lower, upper);
        assert_eq!(mock_rng.random_range(lower..=upper), actual_mocked_val);
    }

    #[test]
    fn test_repeated_push_float() {
        let mut mock_rng = MockRng::new();

        let float_vec: Vec<(f64, f64, f64)> = vec![
            (0.5, 0.0, 1.0),
            (-3.14, -10.0, -2.0),
            (0.0, -5.0, 5.0),
            (42.42, 40.0, 50.0),
            (0.00015, 0.0001, 0.0002)
        ];

        let mut actual_targets = Vec::new();
        for (expected, lower, upper) in float_vec.iter().copied() {
            actual_targets.push(mock_rng.push_range_float(expected, lower, upper));
        }

        for (i, (_, lower, upper)) in float_vec.iter().copied().enumerate() {
            assert!((mock_rng.random_range(lower..upper) - actual_targets[i]).abs() <= f64::EPSILON);
        }

        actual_targets.clear();
        for (expected, lower, upper) in float_vec.iter().copied() {
            actual_targets.push(mock_rng.push_range_float(expected, lower, upper));
        }

        for (i, (_, lower, upper)) in float_vec.iter().copied().enumerate() {
            assert!((mock_rng.random_range(lower..=upper) - actual_targets[i]).abs() <= f64::EPSILON);
        }
    }

    #[test_case(0.5, 0.0, 1.0 ; "standard mid range inclusive")]
    #[test_case(0.0, 0.0, 1.0 ; "exact lower bound inclusive")]
    #[test_case(1.0, 0.0, 1.0 ; "exact upper bound inclusive")]
    #[test_case(-2.0, -10.0, -2.0 ; "exact negative upper bound inclusive")]
    #[test_case(5.5, -5.5, 5.5 ; "exact crossing upper bound inclusive")]
    #[test_case(42.4242, 40.0, 50.0 ; "arbitrary positive floats inclusive")]
    fn test_push_float_inclusive(expected: f64, lower: f64, upper: f64) {
        let mut mock_rng = MockRng::new();

        let actual_mocked_val = mock_rng.push_uniform_float_inclusive(expected, lower, upper);
        let uniform = Uniform::new_inclusive(lower, upper).unwrap();

        assert_eq!(mock_rng.sample(uniform), actual_mocked_val);
    }
}
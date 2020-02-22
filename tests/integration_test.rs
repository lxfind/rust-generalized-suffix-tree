use generalized_suffix_tree;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_suffix() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        let s1 = "ABCABXABCD";
        tree.add_string(String::from(s1), '$');
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{} should be a suffix", &s1[i..]);
        }
        assert!(!tree.is_suffix("A"));
        assert!(!tree.is_suffix("BC"));

        let s2 = "XABCDABCA";
        tree.add_string(String::from(s2), '#');
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{} should be a suffix", &s1[i..]);
        }
        for i in 0..s2.len() {
            assert!(tree.is_suffix(&s2[i..]), "{} should be a suffix", &s2[i..]);
        }
        assert!(!tree.is_suffix("BC"));
    }

    #[test]
    fn test_is_substr() {
        let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
        let s1 = "ABCABXABCD";
        tree.add_string(String::from(s1), '$');
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(
                    tree.is_substr(&s1[i..(j + 1)]),
                    "{} should be a substring",
                    &s1[i..(j + 1)]
                );
            }
        }
        assert!(!tree.is_substr("ABD"));
        assert!(!tree.is_substr("XB"));

        let s2 = "XABCDABCA";
        tree.add_string(String::from(s2), '#');
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(
                    tree.is_substr(&s1[i..(j + 1)]),
                    "{} should be a substring",
                    &s1[i..(j + 1)]
                );
            }
        }
        for i in 0..s2.len() {
            for j in i..s2.len() {
                assert!(
                    tree.is_substr(&s2[i..(j + 1)]),
                    "{} should be a substring",
                    &s2[i..(j + 1)]
                );
            }
        }
        assert!(!tree.is_suffix("BC"));
    }

    #[test]
    fn test_longest_common_substring_all() {
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("ABCABD"), '$');
            tree.add_string(String::from("ABDABCA"), '#');
            tree.pretty_print();
            assert_eq!(tree.longest_common_substring_all(), "ABCA");
        }
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"), '$');
            tree.add_string(String::from("CHOOSETHEGREATALBANIANFUTURE"), '#');
            assert_eq!(tree.longest_common_substring_all(), "THEGREATALBANIA");
        }
    }

    #[test]
    fn test_longest_common_substring_with() {
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"), '$');
            let test_str = String::from("CHOOSETHEGREATALBANIANFUTURE");
            assert_eq!(
                tree.longest_common_substring_with(&test_str),
                "THEGREATALBANIA"
            );
            tree.add_string(test_str, '#');
            let test_str = String::from("VOTECHOOSEGREATALBANIATHEFUTURE");
            assert_eq!(
                tree.longest_common_substring_with(&test_str),
                "EGREATALBANIA"
            );
        }
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("HHDBBCIAAE"), '$');
            let test_str = String::from("AAFJEHDAEG");
            assert_eq!(tree.longest_common_substring_with(&test_str).len(), 2);
        }
    }

    fn gen_random_string(len: usize, alphabet_size: usize) -> String {
        let mut s = String::new();
        for _ in 0..len {
            let ch = (rand::random::<u8>() % alphabet_size as u8) + 'A' as u8;
            s.push(ch as char);
        }
        s
    }

    #[test]
    #[ignore]
    fn test_longest_common_substring_cross_check() {
        for _ in 0..10000 {
            let s1 = gen_random_string(100, 10);
            let s2 = gen_random_string(100, 10);
            let result1 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s1.clone(), '$');
                tree.add_string(s2.clone(), '#');
                tree.longest_common_substring_all()
            };
            let result2 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s1.clone(), '$');
                tree.longest_common_substring_with(&s2)
            };
            let result3 = {
                let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
                tree.add_string(s2.clone(), '$');
                tree.longest_common_substring_with(&s1)
            };
            assert_eq!(result1.len(), result2.len());
            assert_eq!(result1.len(), result3.len());
        }
    }

    #[test]
    fn test_longest_common_substring_all_unicode() {
        {
            let mut tree = generalized_suffix_tree::GeneralizedSuffixTree::new();
            tree.add_string(String::from("我们爱在大自然"), '$');
            tree.add_string(String::from("爱大自然里撒欢"), '#');
            //            tree.pretty_print();
            assert_eq!(tree.longest_common_substring_all(), "大自然");
        }
    }
}

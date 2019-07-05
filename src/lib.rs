use std::collections::HashMap;

type NodeID = usize;
type StrID = usize;
type CharType = u8;

// Special nodes.
const ROOT: NodeID = 0;
const SINK: NodeID = 1;
const INVALID: NodeID = std::usize::MAX;

// Terminator character that will get appended to each input string.
// It is assumed that the input will never contain this character.
const TERM_CHAR: CharType = '$' as CharType;

/// This structure represents a slice to a string.
#[derive(Debug, Clone)]
struct MappedSubstring {
    /// Unique ID of the string it's slicing, which can be used to locate the string from the tree's string storage.
    str_id: StrID,

    /// Index of the first character of the slice.
    start: usize,

    /// One past the index of the last character of the slice.
    /// e.g. when `end` is equal to `start`, this is an empty slice.
    /// Note that `end` here always represents a meaningful index, unlike in the original algorithm where a slice could potentially be open-ended.
    /// Such open-endedness allows for online construction of the tree. Here I chose to not support online construction for convenience. It's possible
    /// to support it by changing `end`'s type to `Option<usize>`.
    end: usize,
}

impl MappedSubstring {
    fn new(str_id: StrID, start: usize, end: usize) -> MappedSubstring {
        MappedSubstring { str_id, start, end }
    }

    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Represents a transition from one node to another.
#[derive(Debug, Clone)]
struct Transition {
    /// The slice of the string this transision represents.
    substr: MappedSubstring,

    /// The ID of the target node after transision.
    target_node: NodeID,

    /// Number of strings sharing this transision, which is needed when computing longest common substring.
    share_count: u32,
}

impl Transition {
    /// Split a transition from the middle at string index `split_index` (the character at `split_index` belongs to the second transision) into two transisions.
    /// These two transisions will be connected through the node with ID of `split_node`. The resulting two transisions are returned.
    fn split(&self, split_index: usize, split_node: NodeID) -> (Transition, Transition) {
        let mut trans1 = self.clone();
        trans1.substr.end = split_index;
        trans1.target_node = split_node;

        let mut trans2 = self.clone();
        trans2.substr.start = split_index;

        (trans1, trans2)
    }
}

/// This is a node in the tree. `transitions` represents all the possible transitions from this node to other nodes, stored using a [`HashMap`]. The hashmap is keyed
/// by the first character of each transition for easy lookup.
/// `suffix_link` contains the suffix link of this node (a term used in the context of Ukkonen's algorithm).
#[derive(Debug)]
struct Node {
    transitions: HashMap<u8, Transition>,
    suffix_link: NodeID,
}

impl Node {
    fn new() -> Node {
        Node {
            transitions: HashMap::new(),
            suffix_link: INVALID,
        }
    }

    fn get_suffix(&self) -> NodeID {
        assert!(self.suffix_link != INVALID, "Invalid suffix link");
        self.suffix_link
    }
}

/// A data structure used to store the current state during the Ukkonen's algorithm.
struct ReferencePoint {
    /// The active node.
    node: NodeID,

    /// The current string we are processing.
    str_id: StrID,

    /// The active point.
    index: usize,
}

impl ReferencePoint {
    fn new(node: NodeID, str_id: StrID, index: usize) -> ReferencePoint {
        ReferencePoint {
            node,
            str_id,
            index,
        }
    }
}

/// This is the generalized suffix tree, implemented using Ukkonen's Algorithm.
/// One important modification to the algorithm is that this is no longer an online
/// algorithm, i.e. it only accepts strings fully provided to the suffix tree, instead
/// of being able to stream processing each string. It is not a fundamental limitation and can be supported.
///
/// # Examples
///
/// ```
/// use generalized_suffix_tree::GeneralizedSuffixTree;
/// let mut tree = GeneralizedSuffixTree::new();
/// tree.add_string(String::from("abcdabce"));
/// tree.add_string(String::from("cdefdefg"));
/// println!("{}", tree.is_suffix("bce"));
/// ```
#[derive(Debug)]
pub struct GeneralizedSuffixTree {
    sink_transition: Transition,
    node_storage: Vec<Node>,
    str_storage: Vec<String>,
}

impl GeneralizedSuffixTree {
    pub fn new() -> GeneralizedSuffixTree {
        let mut root = Node::new();
        let mut sink = Node::new();
        root.suffix_link = SINK;
        sink.suffix_link = ROOT;

        let sink_transition = Transition {
            // The length of the sink_transition is set to 1
            // so that it can consume one charachter during canonize.
            substr: MappedSubstring::new(0, 0, 1),
            target_node: ROOT,
            share_count: 0,
        };

        let node_storage: Vec<Node> = vec![root, sink];
        GeneralizedSuffixTree {
            sink_transition,
            node_storage,
            str_storage: vec![],
        }
    }

    /// Add a new string to the generalized suffix tree.
    pub fn add_string(&mut self, mut s: String) {
        // Augment string with a terminator character.
        assert!(
            !s.as_bytes().contains(&TERM_CHAR),
            "String should not contain '{}'",
            TERM_CHAR
        );
        s.push(TERM_CHAR as char);

        self.str_storage.push(s);

        let index = self.str_storage.len() - 1;
        self.process_suffixes(index);
    }

    pub fn longest_common_substring(&self) -> String {
        let mut cur_str: Vec<MappedSubstring> = vec![];
        let mut cur_len = 0;

        let mut longest_str: Vec<MappedSubstring> = vec![];
        let mut longest_len = 0;

        self.longest_common_substring_recursive(
            ROOT,
            self.str_storage.len() as u32,
            &mut cur_str,
            &mut cur_len,
            &mut longest_str,
            &mut longest_len,
        );

        let mut result = String::new();
        for s in longest_str {
            result.push_str(&self.str_storage[s.str_id][s.start..s.end]);
        }
        if !result.is_empty() && result.as_bytes().last().unwrap() == &TERM_CHAR {
            // Remove the terminator character if any.
            result.pop();
        }

        result
    }

    fn longest_common_substring_recursive(
        &self,
        node: NodeID,
        threshold: u32,
        cur_str: &mut Vec<MappedSubstring>,
        cur_len: &mut usize,
        longest_str: &mut Vec<MappedSubstring>,
        longest_len: &mut usize,
    ) {
        for trans in self.node_storage[node].transitions.values() {
            if trans.share_count == threshold {
                (*cur_str).push(trans.substr.clone());
                *cur_len += trans.substr.len();
                self.longest_common_substring_recursive(
                    trans.target_node,
                    threshold,
                    cur_str,
                    cur_len,
                    longest_str,
                    longest_len,
                );
                *cur_len -= trans.substr.len();
                (*cur_str).pop();
            } else {
                let mut local_cur_str = vec![];
                let mut local_cur_len = 0;
                self.longest_common_substring_recursive(
                    trans.target_node,
                    threshold,
                    &mut local_cur_str,
                    &mut local_cur_len,
                    longest_str,
                    longest_len,
                );
            }
        }
        if *cur_len > *longest_len {
            *longest_len = *cur_len;
            *longest_str = (*cur_str).clone();
        }
    }

    /// Checks whether a given string `s` is a suffix in the suffix tree.
    pub fn is_suffix(&self, s: &str) -> bool {
        self.is_suffix_or_substr(s, false)
    }

    /// Checks whether a given string `s` is a substring of any of the strings
    /// in the suffix tree.
    pub fn is_substr(&self, s: &str) -> bool {
        self.is_suffix_or_substr(s, true)
    }

    fn is_suffix_or_substr(&self, s: &str, is_substr: bool) -> bool {
        let mut node = ROOT;
        let mut index = 0;
        let chars = s.as_bytes();
        while index < s.len() {
            let trans = self.find_transition(node, chars[index]);
            match trans {
                None => return false,
                Some(trans) => {
                    let ref_chars = self.str_storage[trans.substr.str_id].as_bytes();
                    for i in trans.substr.start..trans.substr.end {
                        if index == s.len() {
                            return is_substr || ref_chars[i] == TERM_CHAR;
                        }
                        if chars[index] != ref_chars[i] {
                            return false;
                        }
                        index += 1;
                    }
                    node = trans.target_node;
                }
            };
        }

        is_substr || self.node_storage[node].transitions.contains_key(&TERM_CHAR)
    }

    pub fn pretty_print(&self) {
        self.print_recursive(ROOT, 0);
    }

    fn print_recursive(&self, node: NodeID, space_count: u32) {
        for trans in self.node_storage[node].transitions.values() {
            for _ in 0..space_count {
                print!(" ");
            }
            println!(
                "{} ({})",
                &self.str_storage[trans.substr.str_id][trans.substr.start..trans.substr.end],
                trans.share_count
            );
            self.print_recursive(trans.target_node, space_count + 4);
        }
    }

    fn process_suffixes(&mut self, str_id: StrID) {
        let mut active_point = ReferencePoint::new(ROOT, str_id, 0);
        for i in 0..self.str_storage[str_id].len() {
            let mut cur_str = MappedSubstring::new(str_id, active_point.index, i + 1);
            active_point = self.update(active_point.node, &cur_str);
            cur_str.start = active_point.index;
            active_point = self.canonize(active_point.node, &cur_str);
        }
    }

    fn update(&mut self, node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        assert!(!cur_str.is_empty());

        let mut cur_str = cur_str.clone();

        let mut oldr = ROOT;

        let mut split_str = cur_str.clone();
        split_str.end -= 1;

        let last_ch = self.get_char(cur_str.str_id, cur_str.end - 1);

        let mut active_point = ReferencePoint::new(node, cur_str.str_id, cur_str.start);

        let mut r = node;

        let mut is_endpoint = self.test_and_split(node, &split_str, last_ch, &mut r);
        while !is_endpoint {
            let leaf = self.create_node();
            self.node_storage[r].transitions.insert(
                last_ch,
                Transition {
                    substr: MappedSubstring::new(
                        active_point.str_id,
                        cur_str.end - 1,
                        self.str_storage[active_point.str_id].len(),
                    ),
                    target_node: leaf,
                    share_count: 1,
                },
            );
            if oldr != ROOT {
                self.node_storage[oldr].suffix_link = r;
            }
            oldr = r;
            let suffix = self.node_storage[active_point.node].get_suffix();
            active_point = self.canonize(suffix, &split_str);
            split_str.start = active_point.index;
            cur_str.start = active_point.index;
            is_endpoint = self.test_and_split(active_point.node, &split_str, last_ch, &mut r);
        }
        if oldr != ROOT {
            self.node_storage[oldr].suffix_link = active_point.node;
        }
        active_point
    }

    fn test_and_split(
        &mut self,
        node: NodeID,
        split_str: &MappedSubstring,
        ch: CharType,
        r: &mut NodeID,
    ) -> bool {
        if split_str.is_empty() {
            *r = node;
            return self.find_transition(node, ch).is_some();
        }
        let first_ch = self.get_char(split_str.str_id, split_str.start);
        // Need to clone to avoid unnecessary shared borrow.
        let trans = self.find_transition(node, first_ch).unwrap().clone();
        let split_index = trans.substr.start + split_str.len();
        let ref_ch = self.get_char(trans.substr.str_id, split_index);

        if ref_ch == ch {
            *r = node;
            return true;
        }
        *r = self.create_node();
        let (mut trans1, trans2) = trans.split(split_index, *r);
        self.node_storage[*r].transitions.insert(ref_ch, trans2);

        if trans1.substr.str_id != split_str.str_id {
            trans1.share_count += 1;
        }
        trans1.substr = split_str.clone();

        // This will override the old transition, and replace it with the new one.
        self.node_storage[node].transitions.insert(first_ch, trans1);

        false
    }

    fn canonize(&mut self, mut node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        let mut cur_str = cur_str.clone();
        loop {
            if cur_str.is_empty() {
                return ReferencePoint::new(node, cur_str.str_id, cur_str.start);
            }

            let ch = self.get_char(cur_str.str_id, cur_str.start);
            let trans = self.find_transition_mut(node, ch);
            match trans {
                None => break,
                Some(trans) => {
                    if trans.substr.len() > cur_str.len() {
                        break;
                    }
                    let new_start = cur_str.start + trans.substr.len();
                    let new_node = trans.target_node;

                    if node != SINK && trans.substr.str_id != cur_str.str_id {
                        trans.substr =
                            MappedSubstring::new(cur_str.str_id, cur_str.start, new_start);
                        trans.share_count += 1;
                    }
                    cur_str.start = new_start;
                    node = new_node;
                }
            };
        }
        ReferencePoint::new(node, cur_str.str_id, cur_str.start)
    }

    fn create_node(&mut self) -> NodeID {
        let node = Node::new();
        self.node_storage.push(node);

        self.node_storage.len() - 1
    }

    fn find_transition(&self, node: NodeID, ch: CharType) -> Option<&Transition> {
        if node == SINK {
            Some(&self.sink_transition)
        } else {
            self.node_storage[node].transitions.get(&ch)
        }
    }

    fn find_transition_mut(&mut self, node: NodeID, ch: CharType) -> Option<&mut Transition> {
        if node == SINK {
            Some(&mut self.sink_transition)
        } else {
            self.node_storage[node].transitions.get_mut(&ch)
        }
    }

    fn get_char(&self, str_id: StrID, index: usize) -> u8 {
        self.str_storage[str_id].as_bytes()[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_suffix() {
        let mut tree = GeneralizedSuffixTree::new();
        let s1 = "abcabxabcd";
        tree.add_string(String::from(s1));
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{} should be a suffix", &s1[i..]);
        }
        assert!(!tree.is_suffix("a"));
        assert!(!tree.is_suffix("bc"));

        let s2 = "xabcdabca";
        tree.add_string(String::from(s2));
        for i in 0..s1.len() {
            assert!(tree.is_suffix(&s1[i..]), "{} should be a suffix", &s1[i..]);
        }
        for i in 0..s2.len() {
            assert!(tree.is_suffix(&s2[i..]), "{} should be a suffix", &s2[i..]);
        }
        assert!(!tree.is_suffix("bc"));
    }

    #[test]
    fn test_is_substr() {
        let mut tree = GeneralizedSuffixTree::new();
        let s1 = "abcabxabcd";
        tree.add_string(String::from(s1));
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(
                    tree.is_substr(&s1[i..(j + 1)]),
                    "{} should be a substring",
                    &s1[i..(j + 1)]
                );
            }
        }
        assert!(!tree.is_substr("abd"));
        assert!(!tree.is_substr("xb"));

        let s2 = "xabcdabca";
        tree.add_string(String::from(s2));
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
        assert!(!tree.is_suffix("bc"));
    }

    #[test]
    fn test_longest_common_substring() {
        let mut tree = GeneralizedSuffixTree::new();
        tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"));
        tree.add_string(String::from("CHOOSETHEGREATALBANIANFUTURE"));
        tree.pretty_print();
        assert_eq!(tree.longest_common_substring(), "THEGREATALBANIA");
        tree.add_string(String::from("VOTECHOOSEGREATALBANIATHEFUTURE"));
        assert_eq!(tree.longest_common_substring(), "EGREATALBANIA");
    }
}

use std::collections::HashMap;

type NodeID = u32;
type StrID = u32;
type IndexType = u32;
type CharType = u8;

// Special nodes.
const ROOT: NodeID = 0;
const SINK: NodeID = 1;
const LEAF: NodeID = 2;
const INVALID: NodeID = std::u32::MAX;

// Terminator character that will get appended to each input string.
// It is assumed that the input will never contain this character.
const TERM_CHAR: CharType = '$' as CharType;

/// This structure represents a slice to a string.
#[derive(Debug, Clone)]
struct MappedSubstring {
    /// Unique ID of the string it's slicing, which can be used to locate the string from the tree's string storage.
    str_id: StrID,

    /// Index of the first character of the slice.
    start: IndexType,

    /// One past the index of the last character of the slice.
    /// e.g. when `end` is equal to `start`, this is an empty slice.
    /// Note that `end` here always represents a meaningful index, unlike in the original algorithm where a slice could potentially be open-ended.
    /// Such open-endedness allows for online construction of the tree. Here I chose to not support online construction for convenience. It's possible
    /// to support it by changing `end`'s type to `Option<IndexType>`.
    end: IndexType,
}

impl MappedSubstring {
    fn new(str_id: StrID, start: IndexType, end: IndexType) -> MappedSubstring {
        MappedSubstring { str_id, start, end }
    }

    fn is_empty(&self) -> bool {
        self.start == self.end
    }

    fn len(&self) -> IndexType {
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
    fn split(&self, split_index: IndexType, split_node: NodeID) -> (Transition, Transition) {
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
    index: IndexType,
}

impl ReferencePoint {
    fn new(node: NodeID, str_id: StrID, index: IndexType) -> ReferencePoint {
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
        let leaf = Node::new();
        root.suffix_link = SINK;
        sink.suffix_link = ROOT;

        let sink_transition = Transition {
            // The length of the sink_transition is set to 1
            // so that it can consume one charachter during canonize.
            substr: MappedSubstring::new(0, 0, 1),
            target_node: ROOT,
            share_count: 0,
        };

        let node_storage: Vec<Node> = vec![root, sink, leaf];
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

        let str_id = (self.str_storage.len() - 1) as StrID;
        self.process_suffixes(str_id);
    }

    /// Find the longest common substring among all strings in the suffix.
    /// This function can be used when you already have a suffix tree built,
    /// and would need to know the longest commmon substring.
    /// It can be trivially extended to support longest common substring among
    /// `K` strings.
    pub fn longest_common_substring_all(&self) -> String {
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
            result.push_str(self.get_string_slice(s.str_id, s.start, s.end));
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
        cur_len: &mut IndexType,
        longest_str: &mut Vec<MappedSubstring>,
        longest_len: &mut IndexType,
    ) {
        for trans in self.get_node(node).transitions.values() {
            if trans.share_count != threshold {
                continue;
            }
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
        }
        if *cur_len > *longest_len {
            *longest_len = *cur_len;
            *longest_str = (*cur_str).clone();
        }
    }

    /// Find the longest common substring between string `s` and the current suffix.
    /// This function allows us compute this without adding `s` to the suffix.
    pub fn longest_common_substring_with<'a>(&self, s: &'a String) -> &'a str {
        let mut longest_start: IndexType = 0;
        let mut longest_len: IndexType = 0;
        let mut cur_start: IndexType = 0;
        let mut cur_len: IndexType = 0;
        let mut node: NodeID = ROOT;

        let chars = s.as_bytes();
        let mut index = 0;
        let mut active_length = 0;
        while index < chars.len() {
            let trans = self.get_node(node).transitions.get(&chars[index]);
            match trans {
                None => {}
                Some(trans) => {
                    while index != chars.len()
                        && active_length < trans.substr.len()
                        && self.get_char(trans.substr.str_id, active_length + trans.substr.start)
                            == chars[index]
                    {
                        index += 1;
                        active_length += 1;
                    }

                    let final_len = cur_len + active_length;
                    if final_len > longest_len {
                        longest_len = final_len;
                        longest_start = cur_start;
                    }

                    if index == chars.len() {
                        break;
                    }

                    if active_length == trans.substr.len() {
                        // We can keep following this route.
                        node = trans.target_node;
                        cur_len = final_len;
                        active_length = 0;
                        continue;
                    }
                }
            };
            // There was a mismatch.
            cur_start += 1;
            if cur_start > index as IndexType {
                index += 1;
                continue;
            }
            // We want to follow a different path with one less character from the start.
            let suffix_link = self.get_node(node).suffix_link;
            if suffix_link != INVALID && suffix_link != SINK {
                assert!(cur_len > 0);
                node = suffix_link;
                cur_len -= 1;
            } else {
                node = ROOT;
                active_length = active_length + cur_len - 1;
                cur_len = 0;
            }
            while active_length > 0 {
                let trans = self
                    .get_node(node)
                    .transitions
                    .get(&chars[(cur_start + cur_len) as usize])
                    .unwrap();
                if active_length < trans.substr.len() {
                    break;
                }
                active_length -= trans.substr.len();
                cur_len += trans.substr.len();
                node = trans.target_node;
            }
        }

        &s[longest_start as usize..(longest_start + longest_len) as usize]
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
                    for i in trans.substr.start..trans.substr.end {
                        if index == s.len() {
                            return is_substr || self.get_char(trans.substr.str_id, i) == TERM_CHAR;
                        }
                        if chars[index] != self.get_char(trans.substr.str_id, i) {
                            return false;
                        }
                        index += 1;
                    }
                    node = trans.target_node;
                }
            };
        }

        is_substr || self.get_node(node).transitions.contains_key(&TERM_CHAR)
    }

    pub fn pretty_print(&self) {
        self.print_recursive(ROOT, 0);
    }

    fn print_recursive(&self, node: NodeID, space_count: u32) {
        for trans in self.get_node(node).transitions.values() {
            for _ in 0..space_count {
                print!(" ");
            }
            println!(
                "{} ({})",
                self.get_string_slice(trans.substr.str_id, trans.substr.start, trans.substr.end),
                trans.share_count
            );
            self.print_recursive(trans.target_node, space_count + 4);
        }
    }

    fn process_suffixes(&mut self, str_id: StrID) {
        let mut active_point = ReferencePoint::new(ROOT, str_id, 0);
        for i in 0..self.get_string(str_id).len() {
            let mut cur_str =
                MappedSubstring::new(str_id, active_point.index, (i + 1) as IndexType);
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
            let str_len = self.get_string(active_point.str_id).len() as IndexType;
            self.get_node_mut(r).transitions.insert(
                last_ch,
                Transition {
                    substr: MappedSubstring::new(active_point.str_id, cur_str.end - 1, str_len),
                    target_node: LEAF,
                    share_count: 1,
                },
            );
            if oldr != ROOT {
                self.get_node_mut(oldr).suffix_link = r;
            }
            oldr = r;
            let suffix = self.get_node(active_point.node).get_suffix();
            active_point = self.canonize(suffix, &split_str);
            split_str.start = active_point.index;
            cur_str.start = active_point.index;
            is_endpoint = self.test_and_split(active_point.node, &split_str, last_ch, &mut r);
        }
        if oldr != ROOT {
            self.get_node_mut(oldr).suffix_link = active_point.node;
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
        self.get_node_mut(*r).transitions.insert(ref_ch, trans2);

        if trans1.substr.str_id != split_str.str_id {
            trans1.share_count += 1;
        }
        trans1.substr = split_str.clone();

        // This will override the old transition, and replace it with the new one.
        self.get_node_mut(node).transitions.insert(first_ch, trans1);

        false
    }

    fn canonize(&mut self, mut node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        let mut cur_str = cur_str.clone();
        loop {
            if cur_str.is_empty() {
                return ReferencePoint::new(node, cur_str.str_id, cur_str.start);
            }

            let ch = self.get_char(cur_str.str_id, cur_str.start);
            let prev_node = node;

            {
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
            if node == LEAF {
                // We have just come to a leaf node. Now we need to materialize this shared leaf node storage
                // to a real node, such that we can start extend from it.
                let new_node = self.create_node();
                self.find_transition_mut(prev_node, ch).unwrap().target_node = new_node;
                node = new_node;
            }
        }
        ReferencePoint::new(node, cur_str.str_id, cur_str.start)
    }

    fn create_node(&mut self) -> NodeID {
        let node = Node::new();
        self.node_storage.push(node);

        (self.node_storage.len() - 1) as NodeID
    }

    fn get_node(&self, node_id: NodeID) -> &Node {
        &self.node_storage[node_id as usize]
    }

    fn get_node_mut(&mut self, node_id: NodeID) -> &mut Node {
        assert!(node_id != LEAF, "Cannot modify the shared leaf node");
        &mut self.node_storage[node_id as usize]
    }

    fn get_string(&self, str_id: StrID) -> &String {
        &self.str_storage[str_id as usize]
    }

    fn get_string_slice(&self, str_id: StrID, start: IndexType, end: IndexType) -> &str {
        &self.get_string(str_id)[start as usize..end as usize]
    }

    fn find_transition(&self, node: NodeID, ch: CharType) -> Option<&Transition> {
        if node == SINK {
            Some(&self.sink_transition)
        } else {
            self.get_node(node).transitions.get(&ch)
        }
    }

    fn find_transition_mut(&mut self, node: NodeID, ch: CharType) -> Option<&mut Transition> {
        if node == SINK {
            Some(&mut self.sink_transition)
        } else {
            self.get_node_mut(node).transitions.get_mut(&ch)
        }
    }

    fn get_char(&self, str_id: StrID, index: IndexType) -> u8 {
        self.get_string(str_id).as_bytes()[index as usize]
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
    fn test_longest_common_substring_all() {
        let mut tree = GeneralizedSuffixTree::new();
        tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"));
        tree.add_string(String::from("CHOOSETHEGREATALBANIANFUTURE"));
        assert_eq!(tree.longest_common_substring_all(), "THEGREATALBANIA");
        tree.add_string(String::from("VOTECHOOSEGREATALBANIATHEFUTURE"));
        assert_eq!(tree.longest_common_substring_all(), "EGREATALBANIA");
    }

    #[test]
    fn test_longest_common_substring_with() {
        let mut tree = GeneralizedSuffixTree::new();
        tree.add_string(String::from("VOTEFORTHEGREATALBANIAFORYOU"));
        let test_str = String::from("CHOOSETHEGREATALBANIANFUTURE");
        assert_eq!(
            tree.longest_common_substring_with(&test_str),
            "THEGREATALBANIA"
        );
        tree.add_string(test_str);
        let test_str = String::from("VOTECHOOSEGREATALBANIATHEFUTURE");
        assert_eq!(
            tree.longest_common_substring_with(&test_str),
            "EGREATALBANIA"
        );
    }
}

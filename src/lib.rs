use std::collections::HashMap;

type NodeID = usize;
type StrID = usize;
type CharType = u8;

const ROOT: NodeID = 0;
const SINK: NodeID = 1;
const INVALID: NodeID = std::usize::MAX;

const INFINITY: i32 = std::i32::MAX;

const TERM_CHAR: CharType = '$' as CharType;

// end needs to be signed because it may be -1.
#[derive(Debug, Clone)]
struct MappedSubstring {
    str_id: StrID,
    start: i32,
    end: i32,
}

impl MappedSubstring {
    fn new(str_id: StrID, start: i32, end: i32) -> MappedSubstring {
        MappedSubstring {
            str_id,
            start,
            end,
        }
    }

    fn new_open_end(str_id: StrID, start: i32) -> MappedSubstring {
        MappedSubstring {
            str_id,
            start,
            end: INFINITY,
        }
    }

    fn is_empty(&self) -> bool {
        self.start > self.end
    }

    fn longer_than(&self, another: &MappedSubstring) -> bool {
        self.end - self.start > another.end - another.start
    }

    fn len(&self) -> i32 {
        assert!(self.end != INFINITY, "Unable to get length for open ended substr");
        self.end - self.start + 1
    }
}

#[derive(Debug, Clone)]
struct Transition {
    substr: MappedSubstring,
    target_node: NodeID,
    share_count: u32,
}

impl Transition {
    fn split(&self, split_index: i32, split_node: NodeID) -> (Transition, Transition) {
        let mut trans1 = self.clone();
        trans1.substr.end = split_index - 1;
        trans1.target_node = split_node;

        let mut trans2 = self.clone();
        trans2.substr.start = split_index;

        (trans1, trans2)
    }
}

#[derive(Debug)]
struct Node {
    transitions: HashMap<u8, Transition>,
    suffix_link: NodeID,
}

impl Node {
    fn new_with_suffix(suffix_link: NodeID) -> Node {
        Node {
            transitions: HashMap::new(),
            suffix_link: suffix_link,
        }
    }

    fn new() -> Node {
        Node::new_with_suffix(INVALID)
    }

    fn get_suffix(&self) -> NodeID {
        assert!(self.suffix_link != INVALID, "Invalid suffix link");
        self.suffix_link
    }
}

struct ReferencePoint {
    node: NodeID,
    str_id: StrID,
    index: i32,
}

impl ReferencePoint {
    fn new(node: NodeID, str_id: StrID, index: i32) -> ReferencePoint {
        ReferencePoint {
            node,
            str_id,
            index,
        }
    }
}

#[derive(Debug)]
pub struct GeneralizedSuffixTree {
    sink_transition: Transition,
    node_storage: Vec<Node>,
    str_storage: Vec<String>,
}

impl GeneralizedSuffixTree {
    pub fn new() -> GeneralizedSuffixTree {
        let root = Node::new_with_suffix(SINK);
        let sink = Node::new_with_suffix(ROOT);

        let sink_transition = Transition {
            // The length of the sink_transition is set to 1
            // so that it can consume one charachter during canonize.
            substr: MappedSubstring::new(0, 0, 0),
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

    pub fn add_string(&mut self, mut s: String) {
        // Augment string with a terminator character.
        assert!(
            !s.as_bytes().contains(&TERM_CHAR),
            "String should not contain '{}'",
            TERM_CHAR
        );
        s.push(TERM_CHAR as char);

        self.str_storage.push(s);

        self.process_suffixes(self.str_storage.len() - 1);
    }

    pub fn is_suffix(&self, s: &str) -> bool {
        self.is_suffix_or_substr(s, false)
    }

    pub fn is_substr(&self, s: &str) -> bool {
        self.is_suffix_or_substr(s, true)
    }

    fn is_suffix_or_substr(&self, s: &str, is_substr: bool) -> bool {
        let mut node = ROOT;
        let mut index = 0;
        let chars = s.as_bytes();
        while index < s.len() {
            let trans = self.find_transition(node, chars[index as usize]);
            match trans {
                None => return false,
                Some(trans) => {
                    let ref_chars = self.str_storage[trans.substr.str_id].as_bytes();
                    let trans_end = std::cmp::min(ref_chars.len(), trans.substr.end as usize + 1);
                    for i in trans.substr.start as usize..trans_end {
                        if index == s.len() {
                            return is_substr || ref_chars[i] == TERM_CHAR;
                        }
                        if chars[index] != ref_chars[i] {
                            return false
                        }
                        index += 1;
                    }
                    node = trans.target_node;
                },
            };
        }

        is_substr || self.node_storage[node].transitions.contains_key(&TERM_CHAR)
    }

    fn process_suffixes(&mut self, str_id: StrID) {
        let mut active_point = ReferencePoint::new(ROOT, str_id, 0);
        for i in 0..self.str_storage[str_id].len() {
            let mut cur_str =
                MappedSubstring::new(str_id, active_point.index, i as i32);
            active_point = self.update(active_point.node, &cur_str);
            cur_str.start = active_point.index;
            active_point = self.canonize(active_point.node, &cur_str);
        }
    }

    fn update(&mut self, node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        let mut cur_str = cur_str.clone();

        let mut oldr = ROOT;

        let mut split_str = cur_str.clone();
        split_str.end -= 1;

        let last_ch = self.get_char(cur_str.str_id, cur_str.end);

        let mut active_point = ReferencePoint::new(node, cur_str.str_id, cur_str.start);
        
        let mut r = node;

        let mut is_endpoint = self.test_and_split(node, &split_str, last_ch, &mut r);
        while !is_endpoint {
            let leaf = self.create_node();
            self.node_storage[r].transitions.insert(
                last_ch,
                Transition {
                    substr: MappedSubstring::new_open_end(active_point.str_id, cur_str.end),
                    target_node: leaf,
                    share_count: 1,
                },
            );
            if oldr != ROOT {
                self.node_storage[oldr].suffix_link = r;
            }
            oldr = r;
            active_point = self.canonize(
                self.node_storage[active_point.node].get_suffix(),
                &split_str,
            );
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

        trans1.share_count += 1;
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
            let trans = self.find_transition(node, ch);
            match trans {
                None => break,
                Some(trans) => {
                    if trans.substr.longer_than(&cur_str) {
                        break;
                    }
                    let new_start = cur_str.start + trans.substr.len();
                    let new_node = trans.target_node;

                    if trans.substr.str_id != cur_str.str_id {
                        let new_trans = Transition {
                            substr: MappedSubstring::new(cur_str.str_id, cur_str.start, new_start - 1),
                            target_node: trans.target_node,
                            share_count: trans.share_count + 1,
                        };
                        self.node_storage[node].transitions.insert(ch, new_trans);
                    }
                    cur_str.start = new_start;
                    node = new_node;
                },
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

    fn get_char(&self, str_id: StrID, index: i32) -> u8 {
        self.str_storage[str_id].as_bytes()[index as usize]
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
                assert!(tree.is_substr(&s1[i..(j + 1)]), "{} should be a substring", &s1[i..(j + 1)]);
            }
        }
        assert!(!tree.is_substr("abd"));
        assert!(!tree.is_substr("xb"));

        let s2 = "xabcdabca";
        tree.add_string(String::from(s2));
        for i in 0..s1.len() {
            for j in i..s1.len() {
                assert!(tree.is_substr(&s1[i..(j + 1)]), "{} should be a substring", &s1[i..(j + 1)]);
            }
        }
        for i in 0..s2.len() {
            for j in i..s2.len() {
                assert!(tree.is_substr(&s2[i..(j + 1)]), "{} should be a substring", &s2[i..(j + 1)]);
            }
        }
        assert!(!tree.is_suffix("bc"));
    }
}

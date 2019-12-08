mod disjoint_set;

use std::collections::HashMap;

type NodeID = u32;
type StrID = u32;
type IndexType = u32;
type CharType = u8;

// Special nodes.
const ROOT: NodeID = 0;
const SINK: NodeID = 1;
const INVALID: NodeID = std::u32::MAX;

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

/// This is a node in the tree. `transitions` represents all the possible
/// transitions from this node to other nodes, indexed by the first character
/// of the string slice that transition represents. The character needs to
/// be encoded to an index between 0..MAX_CHAR_COUNT first.
/// `suffix_link` contains the suffix link of this node (a term used in the
/// context of Ukkonen's algorithm).
/// `substr` stores the slice of the string that the transition from the parent
/// node represents. By doing so we avoid having an explicit edge data type.
#[derive(Debug)]
struct Node {
    transitions: HashMap<CharType, NodeID>,

    suffix_link: NodeID,

    /// The slice of the string this node represents.
    substr: MappedSubstring,
}

impl Node {
    fn new(str_id: StrID, start: IndexType, end: IndexType) -> Node {
        Node {
            transitions: HashMap::new(),
            suffix_link: INVALID,
            substr: MappedSubstring::new(str_id, start, end),
        }
    }

    fn get_suffix_link(&self) -> NodeID {
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
/// tree.add_string(String::from("ABCDABCE"), '$');
/// tree.add_string(String::from("CDEFDEFG"), '#');
/// println!("{}", tree.is_suffix("BCE"));
/// ```
#[derive(Debug)]
pub struct GeneralizedSuffixTree {
    node_storage: Vec<Node>,
    str_storage: Vec<String>,
}

impl GeneralizedSuffixTree {
    pub fn new() -> GeneralizedSuffixTree {
        // Set the slice of root to be [0, 1) to allow it consume one character whenever we are transitioning from sink to root.
        // No other node will ever transition to root so this won't affect anything else.
        let mut root = Node::new(0, 0, 1);
        let mut sink = Node::new(0, 0, 0);

        root.suffix_link = SINK;
        sink.suffix_link = ROOT;

        let node_storage: Vec<Node> = vec![root, sink];
        GeneralizedSuffixTree {
            node_storage,
            str_storage: vec![],
        }
    }

    /// Add a new string to the generalized suffix tree.
    pub fn add_string(&mut self, mut s: String, term: char) {
        self.validate_string(&s, term);

        let str_id = self.str_storage.len() as StrID;

        // Add a unique terminator character to the end of the string.
        s.push(term);

        self.str_storage.push(s);
        self.process_suffixes(str_id);
    }

    fn validate_string(&self, s: &str, term: char) {
        assert!(term.is_ascii(), "Only accept ASCII terminator");
        assert!(
            !s.contains(term),
            "String should not contain terminator character"
        );
        for existing_str in &self.str_storage {
            assert!(
                !existing_str.contains(term),
                "Any existing string should not contain terminator character"
            );
        }
    }

    /// Find the longest common substring among all strings in the suffix.
    /// This function can be used when you already have a suffix tree built,
    /// and would need to know the longest commmon substring.
    /// It can be trivially extended to support longest common substring among
    /// `K` strings.
    pub fn longest_common_substring_all(&self) -> String {
        let mut disjoint_set = disjoint_set::DisjointSet::new(self.node_storage.len());

        // prev_node stores the most recent occurance of a leaf that belongs to each string.
        // We use the terminator character (which uniquely represents a string) as the key.
        let mut prev_node: HashMap<CharType, NodeID> = HashMap::new();

        // lca_cnt[v] means the total number of times that the lca of two nodes is node v.
        let mut lca_cnt: Vec<usize> = vec![0; self.node_storage.len()];

        let mut longest_str: (Vec<&MappedSubstring>, IndexType) = (vec![], 0);
        let mut cur_str: (Vec<&MappedSubstring>, IndexType) = (vec![], 0);
        self.longest_common_substring_all_rec(
            &mut disjoint_set,
            &mut prev_node,
            &mut lca_cnt,
            ROOT,
            &mut longest_str,
            &mut cur_str,
        );

        let mut result = String::new();
        for s in longest_str.0 {
            result.push_str(&self.get_string_slice_short(&s));
        }
        result
    }

    /// A recursive DFS that does a couple of things in one run:
    /// - Obtain the each pair of leaves that belong to the same string and are
    ///   consecutive in DFS visits. (stored in `prev_node`)
    /// - Tarjan's Algorithm to compute the least common ancestor for each
    ///   of the above pairs. (information stored in `disjoint_set`)
    /// - Maintain the number of times an LCA lands on each node. (`lca_cnt`)
    /// This function returns a tuple:
    /// - Total number of leaves in the subtree.
    /// - Sum of all LCA counts from each node in the subtree,
    /// including the node itself.
    /// These two numbers can be used to compute the number of unique strings
    /// occured in the subtree, which can be used to check whether we found
    /// a common substring.
    /// Details of the algorithm can be found here:
    /// https://web.cs.ucdavis.edu/~gusfield/cs224f09/commonsubstrings.pdf
    fn longest_common_substring_all_rec<'a>(
        &'a self,
        disjoint_set: &mut disjoint_set::DisjointSet,
        prev_node: &mut HashMap<CharType, NodeID>,
        lca_cnt: &mut Vec<usize>,
        node: NodeID,
        longest_str: &mut (Vec<&'a MappedSubstring>, IndexType),
        cur_str: &mut (Vec<&'a MappedSubstring>, IndexType),
    ) -> (usize, usize) {
        let mut total_leaf = 0;
        let mut total_correction = 0;
        for target_node in self.get_node(node).transitions.values() {
            if *target_node == INVALID {
                continue;
            }
            let slice = &self.get_node(*target_node).substr;
            if slice.end as usize == self.get_string(slice.str_id).len() {
                // target_node is a leaf node.
                total_leaf += 1;
                let last_ch = self.get_char(slice.str_id, slice.end - 1);
                if let Some(prev) = prev_node.get(&last_ch) {
                    let lca = disjoint_set.find_set(*prev as usize);
                    lca_cnt[lca as usize] += 1;
                }
                prev_node.insert(last_ch, *target_node);
            } else {
                cur_str.0.push(slice);
                cur_str.1 += slice.len();
                let result = self.longest_common_substring_all_rec(
                    disjoint_set,
                    prev_node,
                    lca_cnt,
                    *target_node,
                    longest_str,
                    cur_str,
                );
                total_leaf += result.0;
                total_correction += result.1;

                cur_str.0.pop();
                cur_str.1 -= slice.len();
            }

            disjoint_set.union(node as usize, *target_node as usize);
        }
        total_correction += lca_cnt[node as usize];
        let unique_str_cnt = total_leaf - total_correction;
        if unique_str_cnt == self.str_storage.len() {
            // This node represnets a substring that is common among all strings.
            if cur_str.1 > longest_str.1 {
                *longest_str = cur_str.clone();
            }
        }
        (total_leaf, total_correction)
    }

    /// Find the longest common substring between string `s` and the current suffix.
    /// This function allows us compute this without adding `s` to the suffix.
    pub fn longest_common_substring_with<'a>(&self, s: &'a str) -> &'a str {
        let mut longest_start: IndexType = 0;
        let mut longest_len: IndexType = 0;
        let mut cur_start: IndexType = 0;
        let mut cur_len: IndexType = 0;
        let mut node: NodeID = ROOT;

        let chars = s.as_bytes();
        let mut index = 0;
        let mut active_length = 0;
        while index < chars.len() {
            let target_node_id = self.transition(node, chars[index - active_length as usize]);
            if target_node_id != INVALID {
                let slice = &self.get_node(target_node_id).substr;
                while index != chars.len()
                    && active_length < slice.len()
                    && self.get_char(slice.str_id, active_length + slice.start) == chars[index]
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

                if active_length == slice.len() {
                    // We can keep following this route.
                    node = target_node_id;
                    cur_len = final_len;
                    active_length = 0;
                    continue;
                }
            }
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
                assert!(cur_start + cur_len < chars.len() as IndexType);
                let target_node_id = self.transition(node, chars[(cur_start + cur_len) as usize]);
                assert!(target_node_id != INVALID);
                let slice = &self.get_node(target_node_id).substr;
                if active_length < slice.len() {
                    break;
                }
                active_length -= slice.len();
                cur_len += slice.len();
                node = target_node_id;
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

    fn is_suffix_or_substr(&self, s: &str, check_substr: bool) -> bool {
        for existing_str in &self.str_storage {
            assert!(
                !s.contains(existing_str.chars().last().unwrap()),
                "Queried string cannot contain terminator char"
            );
        }
        let mut node = ROOT;
        let mut index = 0;
        let chars = s.as_bytes();
        while index < s.len() {
            let target_node = self.transition(node, chars[index]);
            if target_node == INVALID {
                return false;
            }
            let slice = &self.get_node(target_node).substr;
            for i in slice.start..slice.end {
                if index == s.len() {
                    let is_suffix = i as usize == self.get_string(slice.str_id).len() - 1;
                    return check_substr || is_suffix;
                }
                if chars[index] != self.get_char(slice.str_id, i) {
                    return false;
                }
                index += 1;
            }
            node = target_node;
        }
        let mut is_suffix = false;
        for s in &self.str_storage {
            // The last character of each string is a terminator. We use that
            // to look up in the current transitions to determine if we have
            // reached the end of any string. If needed, we are also able to
            // return which string the queried string is a suffix of.
            if self.transition(node, *s.as_bytes().last().unwrap()) != INVALID {
                is_suffix = true;
                break;
            }
        }

        check_substr || is_suffix
    }

    pub fn pretty_print(&self) {
        self.print_recursive(ROOT, 0);
    }

    fn print_recursive(&self, node: NodeID, space_count: u32) {
        for target_node in self.get_node(node).transitions.values() {
            if *target_node == INVALID {
                continue;
            }
            for _ in 0..space_count {
                print!(" ");
            }
            let slice = &self.get_node(*target_node).substr;
            println!(
                "{}",
                self.get_string_slice(slice.str_id, slice.start, slice.end),
            );
            self.print_recursive(*target_node, space_count + 4);
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
            let leaf_node =
                self.create_node_with_slice(active_point.str_id, cur_str.end - 1, str_len);
            self.set_transition(r, last_ch, leaf_node);
            if oldr != ROOT {
                self.get_node_mut(oldr).suffix_link = r;
            }
            oldr = r;
            let suffix_link = self.get_node(active_point.node).get_suffix_link();
            active_point = self.canonize(suffix_link, &split_str);
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
            return self.transition(node, ch) != INVALID;
        }
        let first_ch = self.get_char(split_str.str_id, split_str.start);

        let target_node_id = self.transition(node, first_ch);
        let target_node_slice = self.get_node(target_node_id).substr.clone();

        let split_index = target_node_slice.start + split_str.len();
        let ref_ch = self.get_char(target_node_slice.str_id, split_index);

        if ref_ch == ch {
            *r = node;
            return true;
        }
        // Split target_node into two nodes by inserting r in the middle.
        *r = self.create_node_with_slice(split_str.str_id, split_str.start, split_str.end);
        self.set_transition(*r, ref_ch, target_node_id);
        self.set_transition(node, first_ch, *r);
        self.get_node_mut(target_node_id).substr.start = split_index;

        false
    }

    fn canonize(&mut self, mut node: NodeID, cur_str: &MappedSubstring) -> ReferencePoint {
        let mut cur_str = cur_str.clone();
        loop {
            if cur_str.is_empty() {
                return ReferencePoint::new(node, cur_str.str_id, cur_str.start);
            }

            let ch = self.get_char(cur_str.str_id, cur_str.start);

            let target_node = self.transition(node, ch);
            if target_node == INVALID {
                break;
            }
            let slice = &self.get_node(target_node).substr;
            if slice.len() > cur_str.len() {
                break;
            }
            cur_str.start += slice.len();
            node = target_node;
        }
        ReferencePoint::new(node, cur_str.str_id, cur_str.start)
    }

    fn create_node_with_slice(
        &mut self,
        str_id: StrID,
        start: IndexType,
        end: IndexType,
    ) -> NodeID {
        let node = Node::new(str_id, start, end);
        self.node_storage.push(node);

        (self.node_storage.len() - 1) as NodeID
    }

    fn get_node(&self, node_id: NodeID) -> &Node {
        &self.node_storage[node_id as usize]
    }

    fn get_node_mut(&mut self, node_id: NodeID) -> &mut Node {
        &mut self.node_storage[node_id as usize]
    }

    fn get_string(&self, str_id: StrID) -> &str {
        &self.str_storage[str_id as usize]
    }

    fn get_string_slice(&self, str_id: StrID, start: IndexType, end: IndexType) -> &str {
        &self.get_string(str_id)[start as usize..end as usize]
    }

    fn get_string_slice_short(&self, slice: &MappedSubstring) -> &str {
        &self.get_string_slice(slice.str_id, slice.start, slice.end)
    }

    fn transition(&self, node: NodeID, ch: CharType) -> NodeID {
        if node == SINK {
            // SINK always transition to ROOT.
            return ROOT;
        }
        match self.get_node(node).transitions.get(&ch) {
            None => INVALID,
            Some(x) => *x,
        }
    }

    fn set_transition(&mut self, node: NodeID, ch: CharType, target_node: NodeID) {
        self.get_node_mut(node).transitions.insert(ch, target_node);
    }

    fn get_char(&self, str_id: StrID, index: IndexType) -> u8 {
        assert!((index as usize) < self.get_string(str_id).len());
        self.get_string(str_id).as_bytes()[index as usize]
    }
}

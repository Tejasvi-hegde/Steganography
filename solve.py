class TreeNode:
     def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        if not root or root == p or root == q :
            return root

        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)

        if left and right :
            return root

        return left if left else right

def main():
    # Create a sample binary tree
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.left = TreeNode(0)
    root.right.right = TreeNode(8)
    
    # Create Solution instance
    solution = Solution()
    
    # Find LCA of nodes with values 5 and 1
    p = root.left.right  # node with value 5
    q = root.right.left  # node with value 1
    lca = solution.lowestCommonAncestor(root, p, q)
    print(f"Lowest Common Ancestor value: {lca.val}")

if __name__ == "__main__":
    main()
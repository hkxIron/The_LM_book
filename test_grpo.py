from GRPO import *

def test_create_completion_mask():
    tokens = torch.Tensor([
        [1,2,3,4,5,10,10],
        [1,2,3,4,10,10,10],
        [10,10,10,10,10,10,10],
        [1,2,3,4,5, 6,  7],
    ])
    """
    mask:
    tensor([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    """
    mask = create_completion_mask(tokens, 10)
    print(mask)

if __name__ == "__main__":
    test_create_completion_mask()
from models.Transformers import *

if __name__ == '__main__':

    print("Building test transformer...")
    test_input = torch.randn(32, 100, 2)
    print("input shape: {}".format(test_input.shape))
    try:
        test_transformer = Transformer(k=2, depth=6, heads=8, mlp_dim=64)
        test_output = test_transformer(test_input)
        print("output shape: {}".format(test_output.shape))
        print("Transformer functional!\n")
    except Exception as e: print(e)

    print("Building vision transformer...")
    img = torch.randn(32, 1, 28, 28)
    print("input shape: {}".format(img.shape))
    try:
        test_transformer = VisionTransformer(image_size=28, patch_size=7, num_classes=10, channels=1,
                               k=64, depth=6, heads=8, mlp_dim=128)
        out = test_transformer(img)
        print("output shape: {}".format(out.shape))
        print("Vision transformer functional!\n")
    except Exception as e: print(e)


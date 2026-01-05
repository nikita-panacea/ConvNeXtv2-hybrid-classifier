# debug_params.py
import importlib, sys, os
sys.path.append(os.getcwd())

def print_model_info(module_name='models.hybrid_model', class_name='HybridConvNeXtV2'):
    import torch
    import inspect
    m = importlib.import_module(module_name)
    ModelClass = getattr(m, class_name)
    print("Instantiating model (pretrained=True)...")
    model = ModelClass(num_classes=8, pretrained=True)

    # total params
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total/1e6:.3f}M")

    # show parameter counts grouped by top-level prefix (first two name parts if available)
    by_prefix = {}
    for name, p in model.named_parameters():
        prefix = name.split('.')[0]
        by_prefix.setdefault(prefix, 0)
        by_prefix[prefix] += p.numel()
    print("\nParameter counts by top-level prefix:")
    for k, v in sorted(by_prefix.items(), key=lambda x: -x[1])[:50]:
        print(f"  {k:30s} : {v/1e6:.3f}M")

    # list largest parameter tensors
    large = [(name, p.numel()) for name, p in model.named_parameters()]
    large.sort(key=lambda x: -x[1])
    print("\nTop 30 largest parameter tensors:")
    for name, cnt in large[:30]:
        print(f"  {name:70s} : {cnt/1e6:.3f}M")

    # Inspect backbone factory if available
    try:
        import models.convnextv2 as cv
        if hasattr(cv, 'convnextv2_tiny'):
            print("\nChecking convnextv2_tiny() standalone size:")
            tiny = cv.convnextv2_tiny(num_classes=1000)
            tcnt = sum(p.numel() for p in tiny.parameters() if p.requires_grad)
            print(f" convnextv2_tiny() params: {tcnt/1e6:.3f}M")
        if hasattr(cv, 'ConvNeXtV2'):
            print("You also have ConvNeXtV2 class available.")
            # try explicit tiny geometry
            tiny_explicit = cv.ConvNeXtV2(in_chans=3, num_classes=1000, depths=[3,3,0,0], dims=[96,192,384,768])
            explicit_cnt = sum(p.numel() for p in tiny_explicit.parameters() if p.requires_grad)
            print(f" ConvNeXtV2(depths=[3,3,0,0], dims=[96,192,384,768]) params: {explicit_cnt/1e6:.3f}M")
    except Exception as e:
        print("Could not inspect convnextv2 module:", e)

if __name__ == '__main__':
    print_model_info()

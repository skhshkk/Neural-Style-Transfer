# ─── 1. Install & Import ─────────────────────────────────────────────
!pip install -q torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ─── 2. Helper: Load & Preprocess Images ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 224  # You can go up to 512 on GPU, but 224 is faster

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)  # shape: [1,3,H,W]
    return img.to(device)

content_img = load_image("/content/modi.jpg")   # upload via files.upload()
style_img   = load_image("/content/modi back.jpg")     # upload via files.upload()

# ─── 3. Load Pretrained VGG & Freeze ─────────────────────────────────
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

# ─── 4. Define Content & Style Layers ───────────────────────────────
content_layers = ["21"]                      # 'conv4_2' in VGG19 - focus on main structure
style_layers   = ["0", "5", "10", "19", "28"] # All style layers for rich texture

# Add more style layers for richer texture transfer
additional_style_layers = ["1", "6", "11", "20"]  # Extra layers for fine details
all_style_layers = style_layers + additional_style_layers

# Function to extract features at these layers
def get_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Precompute content and style features
content_feats = get_features(content_img, vgg, content_layers)
style_feats   = get_features(style_img,   vgg, all_style_layers)  # Use all style layers

# Compute Gram matrices for style
def gram_matrix(tensor):
    # tensor: [batch=1, ch, h, w]
    b, ch, h, w = tensor.size()
    tensor = tensor.view(ch, h * w)
    gram = torch.mm(tensor, tensor.t())
    # Normalize by number of elements
    return gram / (ch * h * w)

style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_feats}# ... existing code ...

# ─── 5. Initialize the Target Image ─────────────────────────────────
# Start with even more noise to allow stronger style adoption
noise = torch.randn_like(content_img) * 0.5  # Increased noise from 0.3 to 0.5
target = (content_img * 0.4 + noise * 0.6).requires_grad_(True)  # More noise influence (was 0.6/0.4)

# ─── 6. Define Optimizer ────────────────────────────────────────────
optimizer = optim.LBFGS([target], lr=1.0)

# ULTRA-STYLE-PROMINENT WEIGHTS - Background will show dominant Van Gogh style!
content_weight = 1e0   # Reduced from 5e0 to 1e0 for less content preservation
style_weight   = 5e4   # Increased from 2e4 to 5e4 for much stronger background style

# ... existing code ...

# ─── 7. Run Optimization Loop ────────────────────────────────────────
iterations = 800  # Increased from 600 to 800 for more thorough style transfer

print("⏳ Starting ULTRA-STYLE-PROMINENT transfer...")
run = [0]
while run[0] <= iterations:
    def closure():
        optimizer.zero_grad()

        # Clamp target to valid range
        target.data.clamp_(0, 1)

        target_feats = get_features(target, vgg, all_style_layers + content_layers)

        # Content loss - preserve main structure
        content_loss = torch.mean(
            (target_feats[content_layers[0]] - content_feats[content_layers[0]]) ** 2
        )

        # Style loss with weighted layers - emphasize different aspects
        style_loss = 0 # Corrected indentation here

        # Primary style layers (original 5) - higher weight with emphasis on early layers
        primary_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # Modified to emphasize early layers even more
        for i, layer in enumerate(style_layers):
            target_gram = gram_matrix(target_feats[layer])
            style_gram  = style_grams[layer]
            layer_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += primary_weights[i] * layer_loss

        # Additional style layers - increased weight for fine texture
        additional_weight = 0.2  # Increased from 0.1 to 0.2
        for layer in additional_style_layers:
            if layer in target_feats and layer in style_grams:
                target_gram = gram_matrix(target_feats[layer])
                style_gram  = style_grams[layer]
                layer_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += additional_weight * layer_loss

        # Total loss - style dominates for prominent background
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        run[0] += 1
        if run[0] % 75 == 0:
            print(f"Iteration {run[0]}/{iterations}")
            print(f"  Total Loss: {total_loss.item():.2f}")
            print(f"  Content Loss: {content_loss.item():.4f}")
            print(f"  Style Loss: {style_loss.item():.4f}")
            style_contribution = (style_weight * style_loss.item())
            content_contribution = (content_weight * content_loss.item())
            print(f"  Style Dominance: {style_contribution/(style_contribution + content_contribution)*100:.1f}%")
            print("-" * 55)

        return total_loss

    optimizer.step(closure)

print("✅ Style transfer complete.")

# ─── 8. Display & Save the Final Image ───────────────────────────────
# Detach and clamp to [0,1]
final_img = target.detach().cpu().squeeze(0)
final_img = torch.clamp(final_img, 0, 1)

# Convert to PIL
unloader = transforms.ToPILImage()
output_image = unloader(final_img)

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original content
content_display = unloader(content_img.cpu().squeeze(0))
axes[0].imshow(content_display)
axes[0].set_title("Content Image")
axes[0].axis("off")

# Style reference
style_display = unloader(style_img.cpu().squeeze(0))
axes[1].imshow(style_display)
axes[1].set_title("Style Image")
axes[1].axis("off")

# Result
axes[2].imshow(output_image)
axes[2].set_title("Style-Prominent Result")
axes[2].axis("off")

plt.tight_layout()
plt.show()

output_image.save("style_prominent_result.png")
print("Saved → style_prominent_result.png")

# ─── 9. Style Prominence Guide ───────────────────────────────────────
print("\n🌟 Ultra Style Prominence Achieved!")
print("Current weights - Content:", content_weight, "Style:", style_weight)
print("Style dominance ratio:", f"{(style_weight/content_weight)*100:.0f}% style influence")
# `
# print("• Want even MORE style? → content_weight=5e-1, style_weight=8e4")
# print("• Too much style? → content_weight=3e0, style_weight=3e4")
# print("• Perfect balance found? → Keep these settings!")

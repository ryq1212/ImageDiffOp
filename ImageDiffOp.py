import os
import numpy as np
from PIL import Image
import torch

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor has 3 channels (RGB)
    assert image.shape[2] == 3, "Input image tensor must have 3 channels (RGB)"
    
    # Normalization
    image = image.float() / 255.0
    
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    max_rgb, argmax_rgb = torch.max(image, dim=2)
    min_rgb = torch.min(image, dim=2)[0]
    delta = max_rgb - min_rgb

    # Hue calculation
    hue = torch.zeros_like(max_rgb)
    
    mask = delta != 0
    mask_r = (argmax_rgb == 0) & mask
    mask_g = (argmax_rgb == 1) & mask
    mask_b = (argmax_rgb == 2) & mask

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    
    hue = hue / 6.0
    hue[~mask] = 0

    # Saturation calculation
    saturation = torch.zeros_like(max_rgb)
    saturation[max_rgb != 0] = delta[max_rgb != 0] / max_rgb[max_rgb != 0]

    # Value calculation
    value = max_rgb

    hsv_image = torch.stack([hue, saturation, value], dim=2)

    # Denormalization
    hsv_image = hsv_image * 255.0

    return hsv_image

# def rgba_to_hsv(image: torch.Tensor) -> torch.Tensor:
#     # Ensure the input tensor has 4 channels (RGBA)
#     assert image.shape[2] == 4, "Input image tensor must have 4 channels (RGBA)"
    
#     # Remove the alpha channel
#     rgb_image = image[:, :, :3]  # Keep only the first 3 channels (RGB)
    
#     # Convert RGB to HSV
#     hsv_image = rgb_to_hsv(rgb_image)
    
#     return hsv_image

def main():

    # Construct the file path of input C
    c_path = input("Enter the path of cover image: ")
    dir_path = os.path.dirname(c_path)
    c_name = os.path.basename(c_path)
    c_prename, _ = os.path.splitext(c_name)
    
    # Load cover image C
    try:
        C = Image.open(c_path).convert("RGBA")
        H = Image.open(c_path).convert("HSV")
    except FileNotFoundError:
        print("Error: Could not find cover image in the path.")
        return

    # Convert image to numpy arrays for easier pixel manipulation
    C_data = np.array(C, dtype=np.int16)
    H_data = np.array(H, dtype=np.int16)
    
    # Move C data to PyTorch tensor
    C_tensor = torch.tensor(C_data, device='cuda', dtype=torch.float16)
    H_tensor = torch.tensor(H_data, device='cuda', dtype=torch.float16)

    # Ask if to use customized base color or a specified base image A
    enterRGB_flag = str(input("Enter 1 if using specified color as base, else loading base image: "))
    if enterRGB_flag == '1':

        try:
            a_r = int(input("Enter R channel of base color (0-255): "))
            a_g = int(input("Enter G channel of base color (0-255): "))
            a_b = int(input("Enter B channel of base color (0-255): "))
            if not ((0 <= a_r <= 255) & (0 <= a_g <= 255) & (0 <= a_b <= 255)):
                raise ValueError
        except ValueError:
            print("Error: RGB must be in the range 0 to 255.")
            return
        
        # Initialize RGB channels of base image A to PyTorch tensor
        A_tensor = torch.zeros_like(C_tensor, device='cuda', dtype=torch.float16)
        A_tensor[:, :, 0] = a_r
        A_tensor[:, :, 1] = a_g
        A_tensor[:, :, 2] = a_b

    else:

        try:
            # Construct the file path of input A
            a_path = input("Enter the path of base image: ")
            a_name = os.path.basename(a_path)
            a_prename, _ = os.path.splitext(a_name)

            # Load base image A
            A = Image.open(a_path).convert("RGBA")
        except FileNotFoundError:
            print("Error: Could not find base image in the path.")
            return
    
        # Ensure images are of the same resolution
        if A.size != C.size:
            print("Error: Base and cover images must be of the same resolution.")
            return

        # Convert image to numpy arrays for easier pixel manipulation
        A_data = np.array(A, dtype=np.int16)
    
        # Move A data to PyTorch tensor
        A_tensor = torch.tensor(A_data, device='cuda', dtype=torch.float16)
    
    # User input for opacity P and step N
    try:
        P = int(input("Enter initial opacity (0-255): "))
        N = int(input("Enter step for increasing opacity (1-255): "))
        if not (0 <= P <= 255) or not (1 <= N <= 255):
            raise ValueError
    except ValueError:
        print("Error: Opacity must be an integer in the range 0 to 255, and step must be an integer in the range 1 to 255.")
        return
    P_init = P
    
    # Calculate the normalized opacity
    P_norm = (1 + P) / 256

    # Ask if using saturation limit and its type
    SatLim_flag = str(input("Enter 1 if using relative saturation limit, and enter 2 for absolute saturation limit, else no limit: "))
    if (SatLim_flag == '1') or (SatLim_flag == '2'): 

        # User input saturation limit
        try:
            S = int(input("Enter saturation limit (1-255) (Warning: lower saturation limit may result higher opacity in output): "))
            if not (1 <= S <= 255):
                raise ValueError
        except ValueError:
            print("Error: Saturation limit must be in the range 1 to 255.")
            return
        
        # Create saturation limit tensor
        S_tensor = torch.zeros_like(H_tensor, device='cuda', dtype=torch.float16)
        if SatLim_flag == '1':
            S_tensor[:, :, 1] = H_tensor[:, :, 1] + S
        elif SatLim_flag == '2':
            S_tensor[:, :, 1] = torch.max(H_tensor[:, :, 1] + 1, S * torch.ones_like(H_tensor[:, :, 1]))  # Add extra 1 considering tolerance in hsv conversion

    # Initialize the output image Y and temporary image T
    Y_tensor = torch.zeros_like(C_tensor, dtype=torch.float16)
    T_tensor = torch.zeros_like(C_tensor, dtype=torch.float16)

    # Create a temporary tensor to store the difference
    diff_tensor = C_tensor[:, :, :3] - A_tensor[:, :, :3]
    
    # Create a mask where in A and C all RGB channels are the same
    ac_same_mask = (diff_tensor[:, :, 0] == 0) & (diff_tensor[:, :, 1] == 0) & (diff_tensor[:, :, 2] == 0)

    # Create a mask for pixels where alpha channel of C is zero, i.e. transparent
    c_transp = C_tensor[:, :, 3] == 0
        
    # Combine masks, whereever not necessary to process is true
    unnecessary_mask = c_transp | ac_same_mask

    # Extract the RGB channels:
    Y_rgb = Y_tensor[:, :, :3]
    Y_alpha = Y_tensor[:, :, 3:]
    T_rgb = T_tensor[:, :, :3]

    # Set opacity to 0 where the combined mask is true, i.e. where is not necessary to process
    Y_alpha[unnecessary_mask] = 0

    # Create a tensor to track clamped pixels and initialize state
    clamped_mask = torch.zeros_like(Y_tensor[:, :, 0], dtype=torch.bool)
    clamping_occurred = False

    if torch.any(~unnecessary_mask) == 0:  # Output a transparent image when no necessary-to-process pixel exists

        Y_tensor[:, :, 3] = 0
    else:

        while torch.any(~unnecessary_mask) == 1:  # Only when necessary-to-process pixels exist 

            # Setting initial opacity of to-process area
            Y_alpha[~unnecessary_mask] = P

            # Initialize clamped pixels
            clamped_mask[:, :] = 0

            for k in range(3):  # Process R, G, B channels in every pixel
            
                result = ((C_tensor[:, :, k] - A_tensor[:, :, k]) / P_norm) + A_tensor[:, :, k]
                clamped = (result < 0) | (result > 255)
                clamped_mask |= clamped
                if torch.any(clamped):
                    clamping_occurred = True
                
                # Processed image to temporary tensor ignoring clamping
                T_rgb[:, :, k] = torch.clamp(result, 0, 255)
            
            if (SatLim_flag == '1') or (SatLim_flag == '2'):  # Consider pixels which saturation beyond limit also as clamped
                T_hsv = rgb_to_hsv(T_rgb)
                clamped_mask |= T_hsv[:, :, 1] > S_tensor[:, :, 1]

            # Only update unclamped and necessary-to-process pixels
            to_update_mask = ~unnecessary_mask & ~clamped_mask

            # Expand the mask to match the shape of the RGB channels
            TUmask_rgb = to_update_mask.unsqueeze(-1).expand_as(Y_rgb)  # [h, w, 3]

            # Combine the images using the to-update mask
            Y_rgb[TUmask_rgb] = T_rgb[TUmask_rgb]
            
            # Update the to-process area for next time loop
            unnecessary_mask |= to_update_mask

            if not clamping_occurred or P >= 255:
                break

            # Increase opacity P by step N
            P = min(P + N, 255)
            P_norm = (1 + P) / 256
    
        # Reconstruct the RGBA image
        Y_tensor = torch.cat((Y_rgb, Y_alpha), dim=-1)  # [h, w, 4]

    # Move Y_tensor back to CPU and convert to numpy array
    Y_data = Y_tensor.cpu().numpy().astype(np.uint8)
    
    # Convert Y_data back to an image
    Y = Image.fromarray(Y_data, 'RGBA')
    
    # Construct the file names and paths of outputs
    if SatLim_flag == '1':
        Slim = '_Slim+'
    elif SatLim_flag == '2': 
        Slim = '_Slim.'
    else:
        S = ''
        Slim = ''
    if enterRGB_flag == '1':
        a_prename = f'{a_r:X}{a_g:X}{a_b:X}'
    y_name = f'{c_prename}_diff.{str(P_init)}.{str(N)}{Slim}{str(S)}_{a_prename}.png'
    z_name = f'{c_prename}_trsp.{str(P_init)}.{str(N)}{Slim}{str(S)}_{a_prename}.png'
    y_path = os.path.join(dir_path, f"{y_name}")
    z_path = os.path.join(dir_path, f"{z_name}")

    # Save the output image
    Y.save(y_path)
    print(f"Output image saved as {y_path}")
    
    if str(input("Enter 1 if output transparency mask:" )) == '1':
        # Create Z image with the same transparency as Y
        Y_alpha_data = (255 - Y_alpha).squeeze(-1)
        Z_data = Y_alpha_data.cpu().numpy().astype(np.uint8)
        
        # Convert Z_data to an image
        Z = Image.fromarray(Z_data, 'L')

        # Save the Z image
        Z.save(z_path)
        print(f"Transparency mask saved as {z_path}")

if __name__ == "__main__":
    main()
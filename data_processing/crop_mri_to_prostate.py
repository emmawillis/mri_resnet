import os
import pydicom
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
mri_dir = '60channelMRIs'
mask_dir = 'masks'
output_dir = 'croppedMRIs'
target_shape = (32, 80, 80)  # (z, y, x)
target_description = "t2_spc_rst_axial obl_Prostate"

def extract_patient_and_mri_id(folder_name):
    parts = folder_name.split('-')
    patient_id = parts[0]
    mri_id = parts[1].split('_')[1]
    return patient_id, mri_id

def load_prostate_mask(mask_folder):
    for fname in os.listdir(mask_folder):
        path = os.path.join(mask_folder, fname)
        try:
            dcm = pydicom.dcmread(path)
            if getattr(dcm, 'SeriesDescription', '') == 'Segmentation of prostate':
                return sitk.ReadImage(path)
        except:
            continue
    return None

def get_mask_center(mask_np):
    nonzero = np.argwhere(mask_np)
    if nonzero.size == 0:
        print("⚠️ Empty mask, falling back to center of volume")
        return tuple([s // 2 for s in mask_np.shape])
    return tuple(np.mean(nonzero, axis=0).astype(int))

def load_mri_volume(mri_folder, target_description):
    series_dict = {}
    for fname in os.listdir(mri_folder):
        path = os.path.join(mri_folder, fname)
        try:
            dcm = pydicom.dcmread(path)
            desc = getattr(dcm, 'SeriesDescription', '')
            if desc not in series_dict:
                series_dict[desc] = []
            series_dict[desc].append((dcm.InstanceNumber, dcm))
        except:
            continue

    if target_description not in series_dict:
        print(f"❌ Series '{target_description}' not found in {mri_folder}")
        print("Found series:", list(series_dict.keys()))
        return None, None, None

    series = sorted(series_dict[target_description], key=lambda x: x[0])
    slices = [d.pixel_array for _, d in series]
    filenames = [d.SOPInstanceUID + ".dcm" for _, d in series]
    metadata = [d for _, d in series]
    return np.stack(slices), filenames, metadata

def apply_bias_field_correction(volume_np):
    print("⚙️  Applying N4 bias field correction...")
    image = sitk.GetImageFromArray(volume_np.astype(np.float32))
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(image, mask)
    return sitk.GetArrayFromImage(corrected)

def normalize_intensity(volume_np):
    min_val = np.min(volume_np)
    max_val = np.max(volume_np)
    if max_val > min_val:
        return (volume_np - min_val) / (max_val - min_val)
    else:
        return volume_np * 0  # constant image

def crop_to_shape(volume_np, center, shape):
    zc, yc, xc = center
    dz, dy, dx = shape
    zmin = max(zc - dz//2, 0)
    ymin = max(yc - dy//2, 0)
    xmin = max(xc - dx//2, 0)

    zmax = zmin + dz
    ymax = ymin + dy
    xmax = xmin + dx

    pad = [(0, max(0, zmax - volume_np.shape[0])),
           (0, max(0, ymax - volume_np.shape[1])),
           (0, max(0, xmax - volume_np.shape[2]))]

    volume_np = np.pad(volume_np, pad, mode='constant')
    return volume_np[zmin:zmax, ymin:ymax, xmin:xmax]

def save_volume_as_dicom(volume, metadata, output_folder, filenames):
    os.makedirs(output_folder, exist_ok=True)
    for i, (dcm, fname) in enumerate(zip(metadata, filenames)):
        if i >= volume.shape[0]:
            break

        # Clip to [0,1] and scale to 16-bit
        slice_img = np.clip(volume[i], 0, 1)
        scaled = (slice_img * 65535).astype(np.uint16)

        dcm.PixelData = scaled.tobytes()
        dcm.Rows, dcm.Columns = scaled.shape
        dcm.BitsStored = 16
        dcm.BitsAllocated = 16
        dcm.HighBit = 15
        dcm.PixelRepresentation = 0  # unsigned

        # ✅ Add correct Window Center / Width
        dcm.WindowCenter = 32768  # midpoint of uint16 range
        dcm.WindowWidth = 65535

        dcm.save_as(os.path.join(output_folder, fname))

def save_debug_slice(volume, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # ✅ create folder if not already there
    mid = volume.shape[0] // 2
    mid = volume.shape[0] // 2
    print(f"🧐 Slice {mid} stats — min: {volume[mid].min()}, max: {volume[mid].max()}, mean: {volume[mid].mean()}")
    plt.imsave(os.path.join(output_folder, "center_slice.png"), volume[mid], cmap='gray')


# --- Main ---
print("🧠 Preprocessing MRI volumes to (80,80,32) centered on prostate...")
for folder in tqdm(os.listdir(mri_dir)):
    mri_path = os.path.join(mri_dir, folder)
    output_path = os.path.join(output_dir, folder)

    patient_id, mri_id = extract_patient_and_mri_id(folder)
    mask_folder = os.path.join(mask_dir, f"Prostate-MRI-US-Biopsy-{patient_id}", f"MR_{mri_id}")
    if not os.path.exists(mask_folder):
        print("❌ Couldn't find mask for", folder)
        continue

    mask = load_prostate_mask(mask_folder)
    if mask is None:
        print("❌ Couldn't load prostate segmentation in", mask_folder)
        continue

    mask_np = sitk.GetArrayFromImage(mask)
    center = get_mask_center(mask_np)

    mri_volume, filenames, metadata = load_mri_volume(mri_path, target_description)
    if mri_volume is None:
        continue

    corrected = apply_bias_field_correction(mri_volume)
    cropped = crop_to_shape(corrected, center, target_shape)
    normalized = normalize_intensity(cropped)

    print("📊 Normalized volume stats - min:", normalized.min(), "max:", normalized.max(), "mean:", normalized.mean())
    # save_debug_slice(normalized, output_path)
    save_volume_as_dicom(normalized, metadata, output_path, filenames)


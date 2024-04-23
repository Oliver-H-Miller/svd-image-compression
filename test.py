import matplotlib.pyplot as plt
import numpy as np
import os
import webbrowser
from PIL import Image

# Ask the user for the name of the image file
image_file = input("[↦ ] Enter the name of the image file (with extension): ")
image_file_no_extension = image_file.split(".")[0]
image_file_extension = image_file.split(".")[1]

# Load the image
img = Image.open('images/uncompressed/' + image_file)
imggray = img.convert('LA') # grayscale

# Convert the image to a matrix
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)

# Save our original image to grayscale
# We will use this later to compare to the compressed image
Image.fromarray(imgmat).convert("L").save('images/uncompressed/grayscale_' + image_file)
print("[✓] Grayscale image has been saved to file (images/uncompressed/grayscale_" + image_file + ").")

# SVD decomposition
U, sigma, V = np.linalg.svd(imgmat)

# Write singular values to file
np.savetxt("data/sigma.csv", sigma, delimiter=",")
print("[✓] Singular values have been written to file (data/sigma.csv).")

# get number of singular values present
max_singular_values = len(sigma)
print("[✓] A total of", max_singular_values, "singular values are present in the image.")

sizes = []
x_values = []

def compress_image(i):
    print("[↦ ] Compressing image with", i, "singular values... ", i, "/", max_singular_values)
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    # save to file
    filename = 'images/compressed/' + str(i) + '_' + image_file
    im = Image.fromarray(reconstimg)
    im = im.convert("L")
    # save as JPEG, with quality set to 100 (no JPEG compression)
    im.save(filename, quality=100, subsampling=0, optimize=False, progressive=False)
    # put the file size in the list (in kB)
    sizes.append(os.path.getsize(filename) / 1024)
    x_values.append(i)

range_list = [i for i in range(1, 101)]
while range_list[-1] < max_singular_values:
    range_list.append(min(range_list[-1] + (len(range_list) - 99), max_singular_values))

for i in range_list:
    compress_image(i)

# Create visualization HTML file from template
html_intermed = ""
with open("intermediate/template.html", "r") as file:
    html_intermed = file.read()

# 2) Replace placeholders with actual values
# {{numbers_arr}}, {{filename}}
html_intermed = html_intermed.replace("{{numbers_arr}}", str(range_list))
html_intermed = html_intermed.replace("{{filename}}", image_file)

with open("visualize.html", "w") as file:
    file.write(html_intermed)
    print("[✓] Visualization file has been created (visualize.html).")

# Plot the file size
plt.figure()
plt.scatter(x_values, sizes, c='r', label='Compressed Image Size')
plt.plot(x_values, sizes, label='Trendline')
plt.xlabel("Number of Singular Values")
plt.ylabel("File Size (kB)")
plt.title("File Size vs Number of Singular Values")
plt.show()

print("[✓] Done.")
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('fileInput');
    const nameInput = document.getElementById('nameInput');
    const processBtn = document.getElementById('processBtn');
    const previewContainer = document.getElementById('previewContainer');
    const modalPreviewContainer = document.getElementById('modalPreviewContainer');
    const modalDownloadBtn = document.getElementById('modalDownloadBtn');
    const modalShareBtn = document.getElementById('modalShareBtn');
    const successModal = new bootstrap.Modal(document.getElementById('successModal'));

    let currentImage = null;
    let cropper = null;
    let stream = null;

    // File Upload Handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewContainer.style.display = 'block';
          
                setupImagePreview(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    });

    // Setup Image Preview
    function setupImagePreview(imageUrl) {
        if (cropper) {
            cropper.destroy();
        }

        const img = document.createElement('img');
        img.src = imageUrl;
        img.className = 'img-fluid';
        
        previewContainer.innerHTML = '';
        previewContainer.appendChild(img);

        cropper = new Cropper(img, {
            aspectRatio: 1,
            viewMode: 1,
            autoCropArea: 1,
            responsive: true,
            restore: false
        });

        currentImage = imageUrl;
        updateProcessButton();
    }

    // Name Input Handler
    nameInput.addEventListener('input', updateProcessButton);

    // Update Process Button State
    function updateProcessButton() {
        processBtn.disabled = !currentImage || !nameInput.value.trim();
    }

    // Process Image
    processBtn.addEventListener('click', async function() {
        if (!currentImage || !nameInput.value.trim()) return;

        const loadingSpinner = document.createElement('div');
        loadingSpinner.className = 'spinner-border text-primary';
        loadingSpinner.setAttribute('role', 'status');
        processBtn.disabled = true;
        processBtn.innerHTML = '';
        processBtn.appendChild(loadingSpinner);

        try {
            // Optimize image before sending
            const croppedCanvas = cropper.getCroppedCanvas({
                width: 800, // Max width
                height: 800, // Max height
                imageSmoothingEnabled: true,
                imageSmoothingQuality: 'high'
            });
            
            // Convert to JPEG for smaller file size
            const imageData = croppedCanvas.toDataURL('image/jpeg', 0.9);
            
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    name: nameInput.value.trim()
                })
            });

            const result = await response.json();

            if (result && result.success && result.image) {
                // Preload the image before showing modal
                const img = new Image();
                img.onload = function() {
                    showSuccessModal(result.image);
                    updateRecentImages();
                };
                img.src = result.image;
            } else {
                throw new Error(result.error || 'Invalid response from server');
            }
        } catch (error) {
            console.error('Error processing image:', error);
            alert('An error occurred while processing your image. Please try again.');
        } finally {
            processBtn.innerHTML = 'Create Template';
            processBtn.disabled = false;
        }
    });

    // Show Success Modal
    function showSuccessModal(imageUrl) {
        const modalImg = document.createElement('img');
        modalImg.src = imageUrl;
        modalImg.className = 'img-fluid';
        modalImg.alt = 'Processed selfie';
        
        modalPreviewContainer.innerHTML = '';
        modalPreviewContainer.appendChild(modalImg);
        
        modalDownloadBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = 'selfie.png';
            link.click();
        };

        modalShareBtn.onclick = async () => {
            try {
                const response = await fetch(imageUrl);
                const blob = await response.blob();
                const file = new File([blob], 'selfie.png', { type: 'image/png' });
                
                if (navigator.share) {
                    await navigator.share({
                        files: [file],
                        title: 'My Selfie',
                        text: 'Check out my selfie!'
                    });
                } else {
                    alert('Sharing is not supported on this device/browser.');
                }
            } catch (error) {
                console.error('Error sharing:', error);
                alert('Unable to share the image. You can download it instead.');
            }
        };

        successModal.show();
    }

    // Update Recent Images
    async function updateRecentImages() {
        try {
            const response = await fetch('/recent-images');
            const data = await response.json();
            
            const recentImagesContainer = document.querySelector('.recent-images');
            if (data.images && Array.isArray(data.images)) {
                recentImagesContainer.innerHTML = data.images.map(image => `
                    <div class="recent-image-card">
                        <img src="/static/${image.path}" alt="Processed selfie">
                        <div class="recent-image-info">
                            <h6 title="${image.name}">${image.name}</h6>
                            <small>${image.timestamp}</small>
                        </div>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('Error updating recent images:', error);
        }
    }

    // Initial load of recent images
    updateRecentImages();
});
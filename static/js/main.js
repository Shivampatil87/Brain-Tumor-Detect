/**
 * Brain Tumor Detection System — Main JS
 * Handles: file upload UX, drag & drop, preview, spinner, form validation
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── Navbar scroll effect ──────────────────────────────────────────────────
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 20) {
        navbar.style.boxShadow = '0 4px 40px rgba(0,0,0,0.5)';
      } else {
        navbar.style.boxShadow = 'none';
      }
    });
  }

  // ── Active nav link ───────────────────────────────────────────────────────
  const currentPath = window.location.pathname;
  document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });

  // ── Upload zone: click to open file picker ────────────────────────────────
  const uploadZone = document.getElementById('uploadZone');
  const fileInput  = document.getElementById('fileInput');
  const uploadForm = document.getElementById('uploadForm');
  const previewContainer = document.getElementById('previewContainer');
  const previewImg        = document.getElementById('previewImg');
  const fileNameDisplay   = document.getElementById('fileNameDisplay');
  const submitBtn         = document.getElementById('submitBtn');
  const spinnerOverlay    = document.getElementById('spinnerOverlay');

  if (uploadZone && fileInput) {

    // Click zone → trigger file input
    uploadZone.addEventListener('click', () => fileInput.click());

    // Drag & Drop support
    uploadZone.addEventListener('dragover', e => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    // File input change
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
      }
    });
  }

  /**
   * Validate file and show preview
   * @param {File} file
   */
  function handleFile(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSizeMB = 16;

    // Type validation
    if (!allowed.includes(file.type)) {
      showError('Invalid file type. Please upload a JPG or PNG image.');
      return;
    }

    // Size validation
    if (file.size > maxSizeMB * 1024 * 1024) {
      showError(`File too large. Maximum size is ${maxSizeMB}MB.`);
      return;
    }

    clearError();

    // Assign to file input (for form submission)
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewContainer) {
        previewContainer.style.display = 'block';
        previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    };
    reader.readAsDataURL(file);

    // Show file name
    if (fileNameDisplay) {
      fileNameDisplay.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
      fileNameDisplay.style.display = 'block';
    }

    // Enable submit button
    if (submitBtn) submitBtn.disabled = false;
  }

  // ── Form submission with spinner ──────────────────────────────────────────
  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      if (!fileInput || fileInput.files.length === 0) {
        e.preventDefault();
        showError('Please select an image file before analyzing.');
        return;
      }
      if (spinnerOverlay) spinnerOverlay.classList.add('active');
    });
  }

  // ── Confidence bar animation on result page ───────────────────────────────
  const confBar = document.querySelector('.confidence-bar');
  if (confBar) {
    const targetWidth = confBar.dataset.width || '0';
    // Trigger after short delay for animation to play
    setTimeout(() => {
      confBar.style.width = targetWidth + '%';
    }, 200);
  }

  // ── Contact form mock submit ──────────────────────────────────────────────
  const contactForm = document.getElementById('contactForm');
  if (contactForm) {
    contactForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const btn = contactForm.querySelector('[type="submit"]');
      const originalText = btn.innerHTML;
      btn.innerHTML = '<span>✅</span> Message Sent!';
      btn.disabled = true;
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
        contactForm.reset();
      }, 3000);
    });
  }

  // ── Utility: show error above upload zone ─────────────────────────────────
  function showError(msg) {
    let el = document.getElementById('uploadError');
    if (!el) {
      el = document.createElement('div');
      el.id = 'uploadError';
      el.className = 'alert-error';
      uploadZone && uploadZone.parentNode.insertBefore(el, uploadZone);
    }
    el.innerHTML = `<span>⚠️</span> ${msg}`;
    el.style.display = 'flex';
  }

  function clearError() {
    const el = document.getElementById('uploadError');
    if (el) el.style.display = 'none';
  }

  // ── Utility: format bytes ─────────────────────────────────────────────────
  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

});

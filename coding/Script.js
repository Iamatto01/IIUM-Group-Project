/* ===== MAIN SCRIPT ===== */
document.addEventListener("DOMContentLoaded", function () {
  // Initialize Leaflet map if element exists
  const mapEl = document.getElementById('map');
  if (mapEl) {
    const staticLat = 3.248447;
    const staticLng = 101.740143;
    const map = L.map('map').setView([staticLat, staticLng], 15);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
    }).addTo(map);
    
    const foodIcon = L.icon({
      iconUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23047857"><path d="M12,12A4,4 0 0,0 16,8A4,4 0 0,0 12,4A4,4 0 0,0 8,8A4,4 0 0,0 12,12M12,13C8.13,13 5,14.57 5,16.71V20H19V16.71C19,14.57 15.87,13 12,13Z" /></svg>',
      iconSize: [40, 40],
      iconAnchor: [20, 40],
      popupAnchor: [0, -40]
    });
    
    L.marker([staticLat, staticLng], {icon: foodIcon})
      .addTo(map)
      .bindPopup(`<strong>Rezki Food</strong><br>Kafeteria Mahallah Al-Faruq, UIAM, Gombak, Malaysia`)
      .openPopup();
  }
  
  // Navigation toggle for mobile
  $('.nav-toggle').click(function() {
    $('nav').toggleClass('active');
  });
  
  // Header scroll effect
  $(window).scroll(function() {
    if ($(this).scrollTop() > 100) {
      $('header').addClass('scrolled');
    } else {
      $('header').removeClass('scrolled');
    }
  });
  
  // Smooth scrolling for anchor links
  $('a[href*="#"]').on('click', function(e) {
    const targetHash = $(this).attr('href');
    if (targetHash.startsWith('#') && $(targetHash).length) {
      e.preventDefault();
      $('html, body').animate({
        scrollTop: $($(this).attr('href')).offset().top - 80
      }, 600, 'swing');
    }
  });
});

/* ===== HOME PAGE SCRIPT ===== */
document.addEventListener("DOMContentLoaded", function() {
  // Mobile nav toggle
  const navToggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('nav');
  if (navToggle && nav) {
    navToggle.addEventListener('click', () => {
      nav.classList.toggle('active');
    });
  }
  
  // Animate nav-grid items on load
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach((item, index) => {
    item.style.opacity = '0';
    item.style.transform = 'translateY(20px)';
    const delay = 100 + index * 100;
    setTimeout(() => {
      item.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      item.style.opacity = '1';
      item.style.transform = 'translateY(0)';
    }, delay);
  });
});

/* ===== MENU PAGE SCRIPT ===== */
document.addEventListener("DOMContentLoaded", function() {
  // Menu filtering functionality
  const categoryBtns = document.querySelectorAll('.category-btn');
  const menuItems = document.querySelectorAll('.menu-item');
  
  if (categoryBtns.length) {
    categoryBtns.forEach(btn => {
      btn.addEventListener('click', function() {
        // Update active button
        categoryBtns.forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        // Filter menu items
        const category = this.dataset.category;
        menuItems.forEach(item => {
          if (category === 'all' || item.dataset.category === category) {
            item.style.display = 'block';
          } else {
            item.style.display = 'none';
          }
        });
      });
    });
  }
});

/* ===== CONTACT PAGE SCRIPT ===== */
document.addEventListener("DOMContentLoaded", function() {
  // Form submission handling
  const contactForm = document.getElementById('contactForm');
  if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Simple validation
      let isValid = true;
      const inputs = this.querySelectorAll('input, textarea');
      
      inputs.forEach(input => {
        if (!input.value.trim()) {
          isValid = false;
          input.style.borderColor = 'red';
        } else {
          input.style.borderColor = '#ddd';
        }
      });
      
      if (isValid) {
        // Show success message (in a real app, would send to server)
        alert('Thank you for your message! We will contact you soon.');
        this.reset();
      } else {
        alert('Please fill in all required fields.');
      }
    });
  }
});

/* ===== LOCATION PAGE SCRIPT ===== */
function openGoogleMaps() {
  const lat = 3.248447;
  const lng = 101.740143;
  const url = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
  window.open(url, '_blank');
}
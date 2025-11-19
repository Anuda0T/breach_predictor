// CyberShield AI - Professional Cybersecurity Web Application JavaScript

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize Application
function initializeApp() {
    setupNavigation();
    setupMobileMenu();
    setupEventListeners();
    showPage('home');
    initializeCharts();
}

// Navigation Setup
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const pageId = this.getAttribute('data-page');
            navigateToPage(pageId);
        });
    });
}

// Mobile Menu Setup
function setupMobileMenu() {
    const mobileToggle = document.getElementById('mobileMenuToggle');
    const navMenu = document.getElementById('navMenu');
    
    mobileToggle.addEventListener('click', function() {
        navMenu.classList.toggle('active');
        this.classList.toggle('active');
    });
}

// Event Listeners Setup
function setupEventListeners() {
    // Risk assessment form
    const riskForm = document.getElementById('riskForm');
    if (riskForm) {
        riskForm.addEventListener('submit', function(e) {
            e.preventDefault();
            calculateRisk();
        });
    }
    
    // Alert filters
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            filterButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            filterAlerts(this.textContent.toLowerCase());
        });
    });
}

// Page Navigation
function navigateToPage(pageId) {
    // Update navigation active state
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    const activeLink = document.querySelector(`[data-page="${pageId}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
    
    // Show page content
    showPage(pageId);
    
    // Update URL hash
    window.location.hash = pageId;
    
    // Close mobile menu
    document.getElementById('navMenu').classList.remove('active');
    document.getElementById('mobileMenuToggle').classList.remove('active');
}

// Show Page Content
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId + '-page');
    if (targetPage) {
        targetPage.classList.add('active');
        
        // Add fade-in animation
        targetPage.style.animation = 'none';
        setTimeout(() => {
            targetPage.style.animation = 'fadeIn 0.6s ease-out';
        }, 10);
    }
    
    // Update page title
    updatePageTitle(pageId);
}

// Update Page Title
function updatePageTitle(pageId) {
    const titles = {
        'home': 'Home - CyberShield AI',
        'threat-monitoring': 'Threat Monitoring - CyberShield AI',
        'risk-assessment': 'Risk Assessment - CyberShield AI',
        'predictions': 'AI Predictions - CyberShield AI',
        'alerts': 'Security Alerts - CyberShield AI',
        'reports': 'Analytics & Reports - CyberShield AI'
    };
    
    document.title = titles[pageId] || 'CyberShield AI';
}

// Risk Assessment Calculator
function calculateRisk() {
    const companyName = document.querySelector('input[placeholder="Enter company name"]').value;
    const industry = document.querySelector('select').value;
    const dataSensitivity = parseInt(document.querySelector('select').nextElementSibling.value);
    const securityBudget = parseFloat(document.querySelector('input[placeholder="Annual budget ($)"]').value) || 0;
    const employeeCount = parseInt(document.querySelector('input[placeholder="Number of Employees"]').value) || 1;
    
    if (!companyName) {
        alert('Please enter a company name');
        return;
    }
    
    // Calculate risk score (simplified algorithm)
    let riskScore = 5.0; // Base risk
    
    // Industry risk factors
    const industryRisk = {
        'Technology': 0.8,
        'Finance': 1.2,
        'Healthcare': 1.5,
        'Education': 0.7,
        'Retail': 0.6,
        'Government': 1.0
    };
    
    riskScore += industryRisk[industry] || 1.0;
    
    // Data sensitivity factor
    riskScore += (dataSensitivity - 1) * 0.5;
    
    // Budget per employee factor
    const budgetPerEmployee = securityBudget / employeeCount;
    if (budgetPerEmployee < 1000) riskScore += 1.0;
    else if (budgetPerEmployee < 2000) riskScore += 0.5;
    
    // Clamp risk score
    riskScore = Math.max(1.0, Math.min(10.0, riskScore));
    
    // Display results
    displayRiskResults(riskScore, companyName);
}

// Display Risk Results
function displayRiskResults(riskScore, companyName) {
    const resultsDiv = document.getElementById('riskResults');
    const riskScoreElement = document.getElementById('riskScore');
    const riskLevelElement = document.getElementById('riskLevel');
    
    // Update risk score
    riskScoreElement.textContent = riskScore.toFixed(1);
    
    // Update risk level
    let riskLevel, riskColor;
    if (riskScore < 3) {
        riskLevel = 'LOW RISK';
        riskColor = '#00d26a';
    } else if (riskScore < 7) {
        riskLevel = 'MEDIUM RISK';
        riskColor = '#ffaa00';
    } else {
        riskLevel = 'HIGH RISK';
        riskColor = '#ff3860';
    }
    
    riskLevelElement.textContent = riskLevel;
    riskLevelElement.style.color = riskColor;
    
    // Show results
    resultsDiv.style.display = 'grid';
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Add success animation
    resultsDiv.style.animation = 'none';
    setTimeout(() => {
        resultsDiv.style.animation = 'fadeIn 0.8s ease-out';
    }, 10);
}

// Filter Alerts
function filterAlerts(filterType) {
    const alertCards = document.querySelectorAll('.alert-card');
    
    alertCards.forEach(card => {
        if (filterType === 'all (47)') {
            card.style.display = 'block';
        } else {
            const severity = filterType.split(' ')[0];
            if (card.classList.contains(`alert-${severity}`)) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        }
    });
}

// Initialize Charts (Placeholder for Chart.js integration)
function initializeCharts() {
    // This would initialize Chart.js charts
    // For now, just log that charts are ready
    console.log('Charts initialized');
    
    // Example: Initialize prediction chart
    const chartContainer = document.getElementById('predictionChart');
    if (chartContainer) {
        // Chart.js integration would go here
        chartContainer.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted);">Chart visualization would be rendered here with Chart.js</div>';
    }
}

// Utility Functions

// Show loading state
function showLoading(element) {
    element.innerHTML = '<div class="loading"></div>';
}

// Hide loading state
function hideLoading(element, content) {
    element.innerHTML = content;
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Debounce function for search inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Add scroll animations
function addScrollAnimations() {
    const animatedElements = document.querySelectorAll('.fade-in');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });
    
    animatedElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(element);
    });
}

// Initialize scroll animations on page load
window.addEventListener('load', addScrollAnimations);

// Handle browser back/forward buttons
window.addEventListener('hashchange', function() {
    const pageId = window.location.hash.substring(1) || 'home';
    navigateToPage(pageId);
});

// Handle initial page load with hash
if (window.location.hash) {
    const initialPage = window.location.hash.substring(1);
    navigateToPage(initialPage);
}

// Export functionality (placeholder)
function exportData(format) {
    alert(`Exporting data as ${format.toUpperCase()}...`);
    // In a real application, this would trigger a download
}

// Print functionality
function printReport() {
    window.print();
}

// Notification system (placeholder)
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search (placeholder)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        // Open search modal
        console.log('Search shortcut triggered');
    }
    
    // Escape key to close modals
    if (e.key === 'Escape') {
        // Close any open modals
        console.log('Escape key pressed');
    }
});

// Performance monitoring
function measurePerformance() {
    if ('performance' in window) {
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
    }
}

// Measure performance on load
window.addEventListener('load', measurePerformance);

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // In production, send to error tracking service
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    // In production, send to error tracking service
});

// Service Worker registration (for PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // navigator.serviceWorker.register('/sw.js')
        //     .then(registration => console.log('SW registered'))
        //     .catch(error => console.log('SW registration failed'));
    });
}

// Accessibility improvements
function setupAccessibility() {
    // Add ARIA labels where needed
    const buttons = document.querySelectorAll('button:not([aria-label])');
    buttons.forEach(button => {
        if (button.textContent.trim()) {
            button.setAttribute('aria-label', button.textContent.trim());
        }
    });
    
    // Focus management
    const focusableElements = document.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid var(--primary-color)';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = 'none';
        });
    });
}

// Initialize accessibility features
setupAccessibility();

// Console welcome message
console.log(`
🛡️ CyberShield AI - Professional Cybersecurity Platform
=======================================================
Features:
• Real-time threat monitoring
• AI-powered risk assessment
• Predictive breach forecasting
• Enterprise security analytics
• Interactive data visualizations

Navigate using the menu above or keyboard shortcuts.
For support, contact: security@cybershield.ai
`);

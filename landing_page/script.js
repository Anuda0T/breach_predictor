// AI-Powered Data Breach Prediction System - Landing Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeLandingPage();
});

function initializeLandingPage() {
    setupAnimations();
    setupInteractiveElements();
    updateSystemStatus();
    setupScrollEffects();
}

// Setup animations and transitions
function setupAnimations() {
    // Add fade-in animation to feature cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe all feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        observer.observe(card);
    });

    // Observe other sections
    document.querySelectorAll('.system-status, .dashboard-preview').forEach(section => {
        observer.observe(section);
    });
}

// Setup interactive elements
function setupInteractiveElements() {
    // Add hover effects to feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // CTA button interaction
    const ctaButton = document.querySelector('.cta-button');
    if (ctaButton) {
        ctaButton.addEventListener('click', function(e) {
            // Add loading state
            this.innerHTML = '🔗 Connecting...';
            this.style.pointerEvents = 'none';

            // Simulate connection delay
            setTimeout(() => {
                this.innerHTML = 'Access Dashboard';
                this.style.pointerEvents = 'auto';
            }, 2000);
        });
    }

    // Feature status indicators
    document.querySelectorAll('.feature-status').forEach(status => {
        status.addEventListener('click', function() {
            this.textContent = this.textContent === '✓' ? '●' : '✓';
            this.style.transform = 'scale(1.2)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 200);
        });
    });
}

// Update system status with simulated data
function updateSystemStatus() {
    // Simulate real-time status updates
    setInterval(() => {
        // Randomly update metrics (for demo purposes)
        const metrics = document.querySelectorAll('.metric-value');

        metrics.forEach(metric => {
            if (Math.random() < 0.1) { // 10% chance to update
                if (metric.classList.contains('operational')) {
                    metric.textContent = Math.random() < 0.95 ? 'Operational' : 'Training';
                } else if (metric.classList.contains('active')) {
                    metric.textContent = Math.random() < 0.98 ? 'Active' : 'Scanning';
                } else if (metric.classList.contains('ready')) {
                    metric.textContent = Math.random() < 0.97 ? 'Ready' : 'Processing';
                }
            }
        });
    }, 5000); // Update every 5 seconds
}

// Setup scroll effects
function setupScrollEffects() {
    let lastScrollY = window.scrollY;

    window.addEventListener('scroll', () => {
        const currentScrollY = window.scrollY;

        // Parallax effect for hero section
        const hero = document.querySelector('.hero');
        if (hero) {
            const speed = 0.5;
            hero.style.transform = `translateY(${currentScrollY * speed}px)`;
        }

        // Update scroll direction for potential animations
        lastScrollY = currentScrollY;
    });
}

// Dashboard mockup interactions
function setupDashboardMockup() {
    const sidebarItems = document.querySelectorAll('.sidebar-item');

    sidebarItems.forEach(item => {
        item.addEventListener('click', function() {
            // Remove active class from all items
            sidebarItems.forEach(i => i.classList.remove('active'));

            // Add active class to clicked item
            this.classList.add('active');

            // Update mockup content based on selection
            updateMockupContent(this.textContent.trim());
        });
    });
}

function updateMockupContent(selectedItem) {
    const mockupCards = document.querySelector('.mockup-cards');
    const mockupChart = document.querySelector('.mockup-chart .chart-placeholder');

    // Simulate different content based on selection
    const contentMap = {
        '📊 Executive Summary': {
            cards: [
                { title: 'System Health', value: '🟢 Operational', bar: 'healthy' },
                { title: 'Active Alerts', value: '0', subtitle: 'All Clear' },
                { title: 'Risk Score', value: '25%', subtitle: 'Low Risk' }
            ],
            chart: '📈 Risk Trend Analysis'
        },
        '🎯 Risk Prediction': {
            cards: [
                { title: 'Current Risk', value: '🔴 78%', bar: 'high-risk' },
                { title: 'Confidence', value: '92%', subtitle: 'High' },
                { title: 'Last Updated', value: '2 min ago', subtitle: 'Real-time' }
            ],
            chart: '📊 Risk Factor Breakdown'
        },
        '📊 Data Analysis': {
            cards: [
                { title: 'Companies', value: '1,247', bar: 'data' },
                { title: 'Data Quality', value: '94%', subtitle: 'Excellent' },
                { title: 'Processing', value: 'Active', subtitle: 'Real-time' }
            ],
            chart: '📈 Data Processing Metrics'
        },
        '🔍 Model Insights': {
            cards: [
                { title: 'Accuracy', value: '87%', bar: 'accuracy' },
                { title: 'Features', value: '23', subtitle: 'Active' },
                { title: 'Last Trained', value: '1 day ago', subtitle: 'Fresh' }
            ],
            chart: '🤖 Feature Importance Analysis'
        }
    };

    const content = contentMap[selectedItem];
    if (content) {
        // Update cards
        const cards = mockupCards.querySelectorAll('.mockup-card');
        content.cards.forEach((cardData, index) => {
            if (cards[index]) {
                const card = cards[index];
                card.querySelector('h4').textContent = cardData.title;

                const valueElement = card.querySelector('.alert-count, .risk-gauge') ||
                                   card.querySelector('span:first-of-type');
                if (valueElement) {
                    valueElement.textContent = cardData.value;
                }

                const subtitle = card.querySelector('span:last-of-type');
                if (subtitle && cardData.subtitle) {
                    subtitle.textContent = cardData.subtitle;
                }

                // Update health bar if present
                const healthBar = card.querySelector('.health-bar');
                if (healthBar && cardData.bar) {
                    healthBar.className = `health-bar ${cardData.bar}`;
                }
            }
        });

        // Update chart
        if (mockupChart) {
            mockupChart.textContent = content.chart;
        }
    }
}

// Performance monitoring
function setupPerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', function() {
        if ('performance' in window) {
            const perfData = performance.getEntriesByType('navigation')[0];
            const loadTime = perfData.loadEventEnd - perfData.loadEventStart;

            console.log(`🚀 Landing page loaded in ${loadTime.toFixed(2)}ms`);

            // Send analytics (simulated)
            if (loadTime > 3000) {
                console.warn('⚠️ Slow load time detected');
            }
        }
    });

    // Monitor user interactions
    let interactionCount = 0;
    document.addEventListener('click', () => {
        interactionCount++;
        if (interactionCount % 10 === 0) {
            console.log(`📊 ${interactionCount} user interactions recorded`);
        }
    });
}

// Error handling
function setupErrorHandling() {
    window.addEventListener('error', function(e) {
        console.error('JavaScript error:', e.error);
        // Could send error reports to monitoring service
    });

    window.addEventListener('unhandledrejection', function(e) {
        console.error('Unhandled promise rejection:', e.reason);
        // Could send error reports to monitoring service
    });
}

// Accessibility enhancements
function setupAccessibility() {
    // Add keyboard navigation for feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.setAttribute('tabindex', '0');

        card.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });

    // Add ARIA labels where needed
    const ctaButton = document.querySelector('.cta-button');
    if (ctaButton) {
        ctaButton.setAttribute('aria-label', 'Access the AI Breach Prediction Dashboard');
    }
}

// Initialize all components
function initializeLandingPage() {
    setupAnimations();
    setupInteractiveElements();
    setupDashboardMockup();
    setupPerformanceMonitoring();
    setupErrorHandling();
    setupAccessibility();
    updateSystemStatus();
    setupScrollEffects();

    // Welcome message
    console.log(`
🛡️ AI-Powered Data Breach Prediction System
===========================================
Landing page loaded successfully!

Features:
• Interactive feature cards with hover effects
• Real-time system status simulation
• Responsive dashboard mockup
• Performance monitoring
• Accessibility support

Navigate using keyboard or mouse for full experience.
    `);
}

// Export functions for potential external use
window.LandingPage = {
    updateSystemStatus,
    updateMockupContent
};

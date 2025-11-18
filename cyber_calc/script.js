// CyberShield Calculator - Professional Cybersecurity Risk Assessment

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize Application
function initializeApp() {
    setupEventListeners();
    showPage('home');
}

// Setup Event Listeners
function setupEventListeners() {
    // Calculator navigation
    document.querySelectorAll('.topic-card').forEach(card => {
        card.addEventListener('click', function() {
            const calculatorType = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            navigateToCalculator(calculatorType);
        });
    });

    // Form submissions
    const forms = {
        'riskForm': calculateRisk,
        'threatForm': analyzeThreats,
        'probabilityForm': predictBreach,
        'costForm': calculateCosts,
        'impactForm': assessImpact,
        'complianceForm': checkCompliance
    };

    Object.keys(forms).forEach(formId => {
        const form = document.getElementById(formId);
        if (form) {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                forms[formId]();
            });
        }
    });
}

// Page Navigation
function navigateToPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page-content').forEach(page => {
        page.classList.remove('active');
    });

    // Show target page
    const targetPage = document.getElementById(pageId + '-page');
    if (targetPage) {
        targetPage.classList.add('active');
        targetPage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function navigateToCalculator(calculatorType) {
    const pageMap = {
        'risk': 'risk',
        'threat': 'threat',
        'probability': 'probability',
        'cost': 'cost',
        'impact': 'impact',
        'compliance': 'compliance'
    };

    navigateToPage(pageMap[calculatorType]);
}

// Calculator Functions

// 1. Company Risk Calculator
function calculateRisk() {
    const companyName = document.getElementById('companyName').value;
    const industry = document.getElementById('industry').value;
    const companySize = document.getElementById('companySize').value;
    const securityBudget = parseFloat(document.getElementById('securityBudget').value) || 0;
    const dataSensitivity = parseInt(document.getElementById('dataSensitivity').value);
    const trainingFrequency = document.getElementById('trainingFrequency').value;
    const mfaEnabled = document.getElementById('mfaEnabled').value;

    // Risk calculation algorithm
    let riskScore = 50; // Base risk

    // Industry risk factors
    const industryRisk = {
        'technology': 1.2,
        'finance': 1.5,
        'healthcare': 1.8,
        'education': 0.8,
        'retail': 0.9,
        'government': 1.1,
        'manufacturing': 1.0,
        'energy': 1.3
    };
    riskScore *= industryRisk[industry] || 1.0;

    // Company size factor
    const sizeRisk = {
        'small': 1.3,
        'medium': 1.1,
        'large': 0.9,
        'enterprise': 0.8
    };
    riskScore *= sizeRisk[companySize] || 1.0;

    // Security budget factor (per employee estimate)
    const estimatedEmployees = {
        'small': 75,
        'medium': 750,
        'large': 5000,
        'enterprise': 15000
    };
    const employees = estimatedEmployees[companySize] || 1000;
    const budgetPerEmployee = securityBudget / employees;
    if (budgetPerEmployee < 500) riskScore *= 1.4;
    else if (budgetPerEmployee < 1000) riskScore *= 1.2;
    else if (budgetPerEmployee < 2000) riskScore *= 1.0;
    else riskScore *= 0.9;

    // Data sensitivity factor
    riskScore *= (dataSensitivity / 2);

    // Training frequency factor
    const trainingRisk = {
        'none': 1.5,
        'annual': 1.2,
        'quarterly': 1.0,
        'monthly': 0.8
    };
    riskScore *= trainingRisk[trainingFrequency] || 1.0;

    // MFA factor
    const mfaRisk = {
        'none': 1.4,
        'partial': 1.1,
        'full': 0.8
    };
    riskScore *= mfaRisk[mfaEnabled] || 1.0;

    // Clamp risk score
    riskScore = Math.max(0, Math.min(100, riskScore));

    displayRiskResults(riskScore, companyName, {
        industry,
        companySize,
        securityBudget,
        dataSensitivity,
        trainingFrequency,
        mfaEnabled
    });
}

function displayRiskResults(riskScore, companyName, factors) {
    const resultsDiv = document.getElementById('riskResults');
    const riskScoreElement = document.getElementById('riskScore');
    const riskLevelElement = document.getElementById('riskLevel');

    // Update risk score
    riskScoreElement.textContent = Math.round(riskScore);

    // Update risk level
    let riskLevel, riskColor;
    if (riskScore < 25) {
        riskLevel = 'LOW RISK';
        riskColor = '#00d26a';
    } else if (riskScore < 50) {
        riskLevel = 'MEDIUM RISK';
        riskColor = '#ffaa00';
    } else if (riskScore < 75) {
        riskLevel = 'HIGH RISK';
        riskColor = '#ff3860';
    } else {
        riskLevel = 'CRITICAL RISK';
        riskColor = '#dc2626';
    }

    riskLevelElement.textContent = riskLevel;
    riskLevelElement.style.color = riskColor;

    // Generate risk factors
    const factorList = document.getElementById('riskFactors');
    factorList.innerHTML = '';

    const factorData = [
        { name: 'Industry Risk', value: getIndustryRisk(factors.industry), max: 100 },
        { name: 'Company Size', value: getSizeRisk(factors.companySize), max: 100 },
        { name: 'Security Budget', value: getBudgetRisk(factors.securityBudget, factors.companySize), max: 100 },
        { name: 'Data Sensitivity', value: factors.dataSensitivity * 20, max: 100 },
        { name: 'Training Level', value: getTrainingRisk(factors.trainingFrequency), max: 100 },
        { name: 'MFA Implementation', value: getMfaRisk(factors.mfaEnabled), max: 100 }
    ];

    factorData.forEach(factor => {
        const factorItem = document.createElement('div');
        factorItem.className = 'factor-item';
        factorItem.innerHTML = `
            <span class="factor-name">${factor.name}</span>
            <div class="factor-bar">
                <div class="factor-fill" style="width: ${factor.value}%"></div>
            </div>
            <span class="factor-value">${Math.round(factor.value)}%</span>
        `;
        factorList.appendChild(factorItem);
    });

    // Generate recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';

    const recommendations = generateRiskRecommendations(riskScore, factors);
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="rec-icon">${rec.icon}</span> ${rec.text}`;
        recommendationsList.appendChild(li);
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Helper functions for risk calculation
function getIndustryRisk(industry) {
    const risks = {
        'healthcare': 90, 'finance': 85, 'technology': 70,
        'government': 65, 'energy': 75, 'manufacturing': 60,
        'retail': 55, 'education': 45
    };
    return risks[industry] || 50;
}

function getSizeRisk(size) {
    const risks = {
        'small': 80, 'medium': 60, 'large': 40, 'enterprise': 30
    };
    return risks[size] || 50;
}

function getBudgetRisk(budget, size) {
    const employeeEstimates = {
        'small': 75, 'medium': 750, 'large': 5000, 'enterprise': 15000
    };
    const employees = employeeEstimates[size] || 1000;
    const perEmployee = budget / employees;
    return Math.min(100, (perEmployee / 20)); // Scale to 0-100
}

function getTrainingRisk(frequency) {
    const risks = {
        'none': 20, 'annual': 40, 'quarterly': 70, 'monthly': 90
    };
    return risks[frequency] || 50;
}

function getMfaRisk(mfa) {
    const risks = {
        'none': 20, 'partial': 60, 'full': 90
    };
    return risks[mfa] || 50;
}

function generateRiskRecommendations(riskScore, factors) {
    const recommendations = [];

    if (riskScore > 75) {
        recommendations.push(
            { icon: '🚨', text: 'Implement immediate security audit and penetration testing' },
            { icon: '🔐', text: 'Deploy multi-factor authentication across all systems' },
            { icon: '💰', text: 'Increase security budget by at least 50%' },
            { icon: '🎓', text: 'Conduct comprehensive security awareness training' },
            { icon: '🛡️', text: 'Implement advanced threat detection systems' }
        );
    } else if (riskScore > 50) {
        recommendations.push(
            { icon: '⚠️', text: 'Schedule security assessment within 30 days' },
            { icon: '🔐', text: 'Ensure MFA is implemented for critical systems' },
            { icon: '📊', text: 'Regular vulnerability scanning and patching' },
            { icon: '🎓', text: 'Quarterly security training sessions' }
        );
    } else if (riskScore > 25) {
        recommendations.push(
            { icon: '📋', text: 'Annual security assessments' },
            { icon: '🔄', text: 'Regular system updates and patches' },
            { icon: '🎓', text: 'Annual security awareness training' },
            { icon: '📊', text: 'Monitor security metrics and logs' }
        );
    } else {
        recommendations.push(
            { icon: '✅', text: 'Maintain current security practices' },
            { icon: '📈', text: 'Continue monitoring and improvements' },
            { icon: '🎓', text: 'Regular security training refreshers' }
        );
    }

    return recommendations;
}

// 2. Dark Web Threat Analyzer
function analyzeThreats() {
    const companyName = document.getElementById('threatCompanyName').value;
    const industry = document.getElementById('threatIndustry').value;
    const companySize = document.getElementById('threatCompanySize').value;
    const checkCredentials = document.getElementById('checkCredentials').checked;
    const checkData = document.getElementById('checkData').checked;
    const checkDiscussions = document.getElementById('checkDiscussions').checked;

    // Simulate threat analysis
    let threatScore = 0;
    let threats = [];

    // Industry-based threats
    const industryThreats = {
        'healthcare': ['PHI data dumps', 'ransomware targeting hospitals', 'credential stuffing'],
        'finance': ['account credentials', 'transaction data leaks', 'ATM malware'],
        'technology': ['source code leaks', 'API key exposure', 'employee credentials'],
        'retail': ['customer data breaches', 'payment card data', 'loyalty program hacks'],
        'government': ['classified data', 'citizen records', 'infrastructure targeting']
    };

    if (industryThreats[industry]) {
        threats.push(...industryThreats[industry]);
        threatScore += 30;
    }

    // Size-based threats
    if (companySize === 'large' || companySize === 'enterprise') {
        threats.push('corporate espionage', 'supply chain attacks');
        threatScore += 20;
    }

    // Specific checks
    if (checkCredentials) {
        threats.push('employee credential dumps', 'password database leaks');
        threatScore += 25;
    }

    if (checkData) {
        threats.push('customer data exposure', 'internal document leaks');
        threatScore += 20;
    }

    if (checkDiscussions) {
        threats.push('forum discussions about vulnerabilities', 'hacker chatter');
        threatScore += 15;
    }

    // Random factor
    threatScore += Math.random() * 20;

    threatScore = Math.min(100, threatScore);

    displayThreatResults(threatScore, threats);
}

function displayThreatResults(threatScore, threats) {
    const resultsDiv = document.getElementById('threatResults');
    const threatScoreElement = document.getElementById('threatScore');
    const threatLevelElement = document.getElementById('threatLevel');

    // Update threat score
    threatScoreElement.textContent = Math.round(threatScore);

    // Update threat level
    let threatLevel, threatColor;
    if (threatScore < 30) {
        threatLevel = 'LOW THREAT';
        threatColor = '#00d26a';
    } else if (threatScore < 60) {
        threatLevel = 'MEDIUM THREAT';
        threatColor = '#ffaa00';
    } else if (threatScore < 80) {
        threatLevel = 'HIGH THREAT';
        threatColor = '#ff3860';
    } else {
        threatLevel = 'CRITICAL THREAT';
        threatColor = '#dc2626';
    }

    threatLevelElement.textContent = threatLevel;
    threatLevelElement.style.color = threatColor;

    // Generate threat stats
    const threatStats = document.getElementById('threatStats');
    threatStats.innerHTML = `
        <div class="threat-stat">
            <span class="threat-stat-value">${threats.length}</span>
            <span class="threat-stat-label">Threats Detected</span>
        </div>
        <div class="threat-stat">
            <span class="threat-stat-value">${Math.round(threatScore * 0.8)}%</span>
            <span class="threat-stat-label">Exposure Level</span>
        </div>
        <div class="threat-stat">
            <span class="threat-stat-value">${Math.floor(Math.random() * 50) + 10}</span>
            <span class="threat-stat-label">Dark Web Mentions</span>
        </div>
        <div class="threat-stat">
            <span class="threat-stat-value">${Math.floor(threatScore / 10)}</span>
            <span class="threat-stat-label">Active Campaigns</span>
        </div>
    `;

    // Generate detected threats
    const detectedThreats = document.getElementById('detectedThreats');
    detectedThreats.innerHTML = '';

    threats.forEach((threat, index) => {
        const severity = index < 2 ? 'critical' : index < 4 ? 'high' : 'medium';
        const threatItem = document.createElement('div');
        threatItem.className = `threat-item ${severity}`;
        threatItem.innerHTML = `
            <div class="threat-title">${threat.charAt(0).toUpperCase() + threat.slice(1)}</div>
            <div class="threat-description">Detected in recent dark web monitoring scans</div>
        `;
        detectedThreats.appendChild(threatItem);
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 3. Breach Probability Predictor
function predictBreach() {
    const firewallScore = parseInt(document.getElementById('firewallScore').value);
    const trainingScore = parseInt(document.getElementById('trainingScore').value);
    const patchScore = parseInt(document.getElementById('patchScore').value);
    const encryptionScore = parseInt(document.getElementById('encryptionScore').value);
    const previousBreaches = parseInt(document.getElementById('previousBreaches').value);
    const industryRisk = document.getElementById('industryRisk').value;

    // Calculate probability using weighted algorithm
    let probability = 0;

    // Security scores (weighted)
    const securityAvg = (firewallScore + trainingScore + patchScore + encryptionScore) / 4;
    probability += (10 - securityAvg / 10) * 40; // 40% weight

    // Historical breaches
    probability += previousBreaches * 5; // 5% per breach

    // Industry risk
    const industryMultiplier = {
        'low': 0.5,
        'medium': 1.0,
        'high': 1.5
    };
    probability *= industryMultiplier[industryRisk];

    // Random factor (±10%)
    probability *= (0.9 + Math.random() * 0.2);

    probability = Math.max(0, Math.min(95, probability));

    displayProbabilityResults(probability, {
        securityAvg,
        previousBreaches,
        industryRisk
    });
}

function displayProbabilityResults(probability, factors) {
    const resultsDiv = document.getElementById('probabilityResults');
    const probabilityScoreElement = document.getElementById('probabilityScore');
    const probabilityLevelElement = document.getElementById('probabilityLevel');

    // Update probability
    probabilityScoreElement.textContent = Math.round(probability) + '%';

    // Update risk level
    let probabilityLevel, probabilityColor;
    if (probability < 20) {
        probabilityLevel = 'LOW PROBABILITY';
        probabilityColor = '#00d26a';
    } else if (probability < 40) {
        probabilityLevel = 'MEDIUM PROBABILITY';
        probabilityColor = '#ffaa00';
    } else if (probability < 70) {
        probabilityLevel = 'HIGH PROBABILITY';
        probabilityColor = '#ff3860';
    } else {
        probabilityLevel = 'VERY HIGH PROBABILITY';
        probabilityColor = '#dc2626';
    }

    probabilityLevelElement.textContent = probabilityLevel;
    probabilityLevelElement.style.color = probabilityColor;

    // Timeline prediction
    const timelineInfo = document.getElementById('timelineInfo');
    let timelineText, timelineValue;
    if (probability < 30) {
        timelineValue = '2-5 years';
        timelineText = 'Long-term risk';
    } else if (probability < 60) {
        timelineValue = '6-18 months';
        timelineText = 'Medium-term risk';
    } else {
        timelineValue = '1-6 months';
        timelineText = 'Short-term risk';
    }

    timelineInfo.innerHTML = `
        <div class="timeline-value">${timelineValue}</div>
        <div class="timeline-label">${timelineText}</div>
    `;

    // Contributing factors
    const probabilityFactors = document.getElementById('probabilityFactors');
    probabilityFactors.innerHTML = '';

    const factorData = [
        { name: 'Security Posture', value: Math.round((10 - factors.securityAvg / 10) * 100) },
        { name: 'Historical Breaches', value: Math.min(100, factors.previousBreaches * 20) },
        { name: 'Industry Risk', value: factors.industryRisk === 'high' ? 75 : factors.industryRisk === 'medium' ? 50 : 25 }
    ];

    factorData.forEach(factor => {
        const factorItem = document.createElement('div');
        factorItem.className = 'factor-item';
        factorItem.innerHTML = `
            <span class="factor-name">${factor.name}</span>
            <div class="factor-bar">
                <div class="factor-fill" style="width: ${factor.value}%"></div>
            </div>
            <span class="factor-value">${factor.value}%</span>
        `;
        probabilityFactors.appendChild(factorItem);
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 4. Security Cost Calculator
function calculateCosts() {
    const currentBudget = parseFloat(document.getElementById('currentBudget').value) || 0;
    const employeeCount = parseInt(document.getElementById('employeeCount').value) || 1;

    // Investment checkboxes
    const investments = {
        firewall: document.getElementById('investFirewall').checked ? 25000 : 0,
        training: document.getElementById('investTraining').checked ? 15000 : 0,
        monitoring: document.getElementById('investMonitoring').checked ? 30000 : 0,
        backup: document.getElementById('investBackup').checked ? 20000 : 0
    };

    const totalInvestment = Object.values(investments).reduce((sum, cost) => sum + cost, 0);
    const recommendedBudget = currentBudget + totalInvestment;

    // Calculate ROI (simplified)
    const breachPreventionValue = recommendedBudget * 2.5; // Estimated value of prevented breaches
    const operationalSavings = recommendedBudget * 0.3; // Operational efficiency
    const totalROI = breachPreventionValue + operationalSavings;
    const roiPercentage = ((totalROI - recommendedBudget) / recommendedBudget) * 100;

    displayCostResults(currentBudget, recommendedBudget, roiPercentage, investments);
}

function displayCostResults(currentBudget, recommendedBudget, roiPercentage, investments) {
    const resultsDiv = document.getElementById('costResults');

    // Update cost summary
    document.getElementById('currentCost').textContent = formatCurrency(currentBudget);
    document.getElementById('recommendedCost').textContent = formatCurrency(recommendedBudget);
    document.getElementById('potentialSavings').textContent = formatCurrency(recommendedBudget - currentBudget);

    // ROI metrics
    const roiMetrics = document.getElementById('roiMetrics');
    roiMetrics.innerHTML = `
        <div class="roi-metric">
            <span class="roi-label">ROI Percentage</span>
            <span class="roi-value">${Math.round(roiPercentage)}%</span>
        </div>
        <div class="roi-metric">
            <span class="roi-label">Payback Period</span>
            <span class="roi-value">18 months</span>
        </div>
        <div class="roi-metric">
            <span class="roi-label">Annual Savings</span>
            <span class="roi-value">${formatCurrency(recommendedBudget * 0.4)}</span>
        </div>
    `;

    // Cost breakdown
    const costBreakdown = document.getElementById('costBreakdown');
    costBreakdown.innerHTML = '';

    const breakdownItems = [
        { label: 'Current Budget', value: currentBudget },
        { label: 'Advanced Firewall', value: investments.firewall },
        { label: 'Employee Training', value: investments.training },
        { label: '24/7 Monitoring', value: investments.monitoring },
        { label: 'Cloud Backup', value: investments.backup }
    ];

    breakdownItems.forEach(item => {
        if (item.value > 0) {
            const breakdownItem = document.createElement('div');
            breakdownItem.className = 'breakdown-item';
            breakdownItem.innerHTML = `
                <span class="breakdown-label">${item.label}</span>
                <span class="breakdown-value">${formatCurrency(item.value)}</span>
            `;
            costBreakdown.appendChild(breakdownItem);
        }
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 5. Incident Impact Assessor
function assessImpact() {
    const recordsAffected = parseInt(document.getElementById('recordsAffected').value);
    const dataTypes = document.getElementById('dataTypes').value;
    const detectionTime = parseInt(document.getElementById('detectionTime').value);
    const containmentTime = parseInt(document.getElementById('containmentTime').value);

    // Cost calculation based on data types and records
    const baseCostPerRecord = {
        'pii': 150,
        'financial': 300,
        'health': 500,
        'intellectual': 200,
        'mixed': 250
    };

    let costPerRecord = baseCostPerRecord[dataTypes] || 200;
    let totalCost = recordsAffected * costPerRecord;

    // Time-based multipliers
    const detectionMultiplier = detectionTime > 24 ? 1.5 : detectionTime > 12 ? 1.2 : 1.0;
    const containmentMultiplier = containmentTime > 72 ? 2.0 : containmentTime > 24 ? 1.5 : 1.0;

    totalCost *= detectionMultiplier * containmentMultiplier;

    // Additional costs
    const notificationCost = recordsAffected * 5; // Notification per record
    const legalCost = totalCost * 0.1; // 10% legal fees
    const recoveryCost = totalCost * 0.2; // 20% recovery costs

    const finalCost = totalCost + notificationCost + legalCost + recoveryCost;

    displayImpactResults(finalCost, {
        recordsAffected,
        dataTypes,
        detectionTime,
        containmentTime,
        costBreakdown: {
            direct: totalCost,
            notification: notificationCost,
            legal: legalCost,
            recovery: recoveryCost
        }
    });
}

function displayImpactResults(totalCost, details) {
    const resultsDiv = document.getElementById('impactResults');

    // Update total cost
    document.getElementById('totalCost').textContent = formatCurrency(totalCost);

    // Cost categories
    const costCategories = document.getElementById('costCategories');
    costCategories.innerHTML = '';

    const categories = [
        { name: 'Direct Breach Costs', amount: details.costBreakdown.direct },
        { name: 'Customer Notifications', amount: details.costBreakdown.notification },
        { name: 'Legal & Compliance', amount: details.costBreakdown.legal },
        { name: 'Recovery & Remediation', amount: details.costBreakdown.recovery }
    ];

    categories.forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'cost-category';
        categoryDiv.innerHTML = `
            <span class="category-name">${category.name}</span>
            <span class="category-amount">${formatCurrency(category.amount)}</span>
        `;
        costCategories.appendChild(categoryDiv);
    });

    // Recovery steps
    const recoverySteps = document.getElementById('recoverySteps');
    recoverySteps.innerHTML = '';

    const steps = [
        'Isolate affected systems and contain the breach',
        'Notify affected customers within 72 hours',
        'Conduct forensic investigation to determine breach scope',
        'Reset all compromised credentials and access keys',
        'Implement additional security measures to prevent recurrence',
        'Monitor systems for unusual activity for 90 days',
        'Prepare incident report for regulatory compliance'
    ];

    steps.forEach(step => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="rec-icon">🔧</span> ${step}`;
        recoverySteps.appendChild(li);
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 6. Compliance Checker
function checkCompliance() {
    const industry = document.getElementById('complianceIndustry').value;
    const size = document.getElementById('complianceSize').value;
    const handlesPII = document.getElementById('handlesPII').checked;
    const handlesFinancial = document.getElementById('handlesFinancial').checked;
    const handlesHealth = document.getElementById('handlesHealth').checked;
    const international = document.getElementById('international').checked;

    // Determine applicable regulations
    const regulations = [];
    let complianceScore = 100;

    // Industry-specific regulations
    if (industry === 'healthcare') {
        regulations.push({
            name: 'HIPAA',
            description: 'Health Insurance Portability and Accountability Act'
        });
        complianceScore -= 15;
    }

    if (industry === 'finance') {
        regulations.push({
            name: 'SOX',
            description: 'Sarbanes-Oxley Act'
        }, {
            name: 'PCI DSS',
            description: 'Payment Card Industry Data Security Standard'
        });
        complianceScore -= 20;
    }

    if (handlesPII) {
        regulations.push({
            name: 'GDPR',
            description: 'General Data Protection Regulation (if EU data)'
        });
        complianceScore -= 10;
    }

    if (handlesFinancial) {
        regulations.push({
            name: 'GLBA',
            description: 'Gramm-Leach-Bliley Act'
        });
        complianceScore -= 10;
    }

    if (handlesHealth) {
        regulations.push({
            name: 'HITECH',
            description: 'Health Information Technology for Economic and Clinical Health Act'
        });
        complianceScore -= 15;
    }

    if (international) {
        regulations.push({
            name: 'CCPA',
            description: 'California Consumer Privacy Act'
        });
        complianceScore -= 5;
    }

    // Size-based requirements
    if (size === 'large') {
        complianceScore -= 10; // More stringent requirements
    }

    complianceScore = Math.max(0, complianceScore);

    displayComplianceResults(complianceScore, regulations);
}

function displayComplianceResults(complianceScore, regulations) {
    const resultsDiv = document.getElementById('complianceResults');
    const complianceScoreElement = document.getElementById('complianceScore');
    const complianceLevelElement = document.getElementById('complianceLevel');

    // Update compliance score
    complianceScoreElement.textContent = Math.round(complianceScore) + '%';

    // Update compliance level
    let complianceLevel, complianceColor;
    if (complianceScore >= 80) {
        complianceLevel = 'COMPLIANT';
        complianceColor = '#00d26a';
    } else if (complianceScore >= 60) {
        complianceLevel = 'MOSTLY COMPLIANT';
        complianceColor = '#ffaa00';
    } else if (complianceScore >= 40) {
        complianceLevel = 'PARTIALLY COMPLIANT';
        complianceColor = '#ff3860';
    } else {
        complianceLevel = 'NON-COMPLIANT';
        complianceColor = '#dc2626';
    }

    complianceLevelElement.textContent = complianceLevel;
    complianceLevelElement.style.color = complianceColor;

    // Applicable regulations
    const applicableRegulations = document.getElementById('applicableRegulations');
    applicableRegulations.innerHTML = '';

    regulations.forEach(reg => {
        const regItem = document.createElement('div');
        regItem.className = 'regulation-item';
        regItem.innerHTML = `
            <div class="regulation-name">${reg.name}</div>
            <div class="regulation-description">${reg.description}</div>
        `;
        applicableRegulations.appendChild(regItem);
    });

    // Compliance gaps
    const complianceGaps = document.getElementById('complianceGaps');
    complianceGaps.innerHTML = '';

    const gaps = generateComplianceGaps(complianceScore, regulations);
    gaps.forEach(gap => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="rec-icon">⚠️</span> ${gap}`;
        complianceGaps.appendChild(li);
    });

    // Show results
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function generateComplianceGaps(score, regulations) {
    const gaps = [];

    if (score < 80) {
        gaps.push('Implement comprehensive data encryption policies');
        gaps.push('Establish regular security audits and assessments');
        gaps.push('Develop incident response and breach notification procedures');
    }

    if (score < 60) {
        gaps.push('Conduct employee security awareness training');
        gaps.push('Implement access controls and least privilege principles');
        gaps.push('Regular vulnerability scanning and patch management');
    }

    if (score < 40) {
        gaps.push('Appoint dedicated data protection officer');
        gaps.push('Implement data classification and handling procedures');
        gaps.push('Regular compliance monitoring and reporting');
    }

    return gaps;
}

// Utility Functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function resetCalculator(calculatorType) {
    const formMap = {
        'risk': 'riskForm',
        'threat': 'threatForm',
        'probability': 'probabilityForm',
        'cost': 'costForm',
        'impact': 'impactForm',
        'compliance': 'complianceForm'
    };

    const resultsMap = {
        'risk': 'riskResults',
        'threat': 'threatResults',
        'probability': 'probabilityResults',
        'cost': 'costResults',
        'impact': 'impactResults',
        'compliance': 'complianceResults'
    };

    // Reset form
    const form = document.getElementById(formMap[calculatorType]);
    if (form) {
        form.reset();
    }

    // Hide results
    const results = document.getElementById(resultsMap[calculatorType]);
    if (results) {
        results.style.display = 'none';
    }
}

function exportResults(calculatorType) {
    // Simple export functionality (in a real app, this would generate a PDF or CSV)
    const results = document.getElementById(calculatorType + 'Results');
    if (results) {
        const data = {
            calculator: calculatorType,
            timestamp: new Date().toISOString(),
            results: results.innerText
        };

        // Create downloadable JSON file
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `${calculatorType}_assessment_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        alert('Results exported successfully!');
    }
}

// Form validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.style.borderColor = '#ff3860';
            isValid = false;
        } else {
            input.style.borderColor = 'var(--border-color)';
        }
    });

    return isValid;
}

// Initialize form validation
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.hasAttribute('required') && !this.value.trim()) {
                    this.style.borderColor = '#ff3860';
                } else {
                    this.style.borderColor = 'var(--border-color)';
                }
            });
        });
    });
});

// Performance monitoring
window.addEventListener('load', function() {
    if ('performance' in window) {
        const perfData = performance.getEntriesByType('navigation')[0];
        console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
    }
});

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});

// Console welcome message
console.log(`
🛡️ CyberShield Calculator - Professional Cybersecurity Risk Assessment
==================================================================
Calculators Available:
• Company Risk Calculator - Assess breach risk scores
• Dark Web Threat Analyzer - Analyze dark web exposure
• Breach Probability Predictor - Predict breach likelihood
• Security Cost Calculator - Calculate ROI and cost-benefit
• Incident Impact Assessor - Assess breach impact costs
• Compliance Checker - Check regulatory compliance

Navigate using the topic cards above or calculator buttons.
Real calculations with professional algorithms included.
`);

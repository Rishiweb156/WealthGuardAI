/**
 * WealthGuard AI - Enhanced Dashboard Application
 * Complete frontend logic for Graph RAG, Forecasting, and AI Chat
 */

// API Configuration
const API_BASE = window.location.origin;

// Global State
const state = {
    transactions: [],
    graphData: null,
    forecastData: null,
    visualizations: null,
    currentFiles: []
};

// Chart instances
let spendingChart = null;
let pieChart = null;
let forecastChart = null;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 WealthGuard AI Dashboard Initialized');
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    loadInitialData();
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-menu li').forEach(li => {
        li.addEventListener('click', (e) => switchTab(e.target.closest('li')));
    });
    
    // Upload Modal
    document.getElementById('upload-btn').addEventListener('click', openUploadModal);
    document.getElementById('close-modal').addEventListener('click', closeUploadModal);
    document.getElementById('cancel-upload').addEventListener('click', closeUploadModal);
    document.getElementById('browse-files').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    document.getElementById('file-input').addEventListener('change', handleFileSelect);
    document.getElementById('submit-upload').addEventListener('click', handleUpload);
    
    // Drag and Drop
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // Refresh
    document.getElementById('refresh-btn').addEventListener('click', loadDashboardData);
    
    // Graph RAG
    document.getElementById('run-graph-analysis').addEventListener('click', runGraphAnalysis);
    
    // Forecast
    document.getElementById('run-forecast').addEventListener('click', runForecast);
    
    // AI Chat
    document.getElementById('send-chat').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    
    // Search Transactions
    document.getElementById('search-transactions')?.addEventListener('input', filterTransactions);
    document.getElementById('filter-category')?.addEventListener('change', filterTransactions);
}

// ==================== TAB SWITCHING ====================
function switchTab(element) {
    // Update nav menu
    document.querySelectorAll('.nav-menu li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    // Get tab name
    const tabName = element.dataset.tab;
    
    // Update page title
    const titles = {
        'dashboard': 'Financial Dashboard',
        'transactions': 'Transactions',
        'graph-insights': 'Graph RAG Analysis',
        'forecast': 'Predictive Forecasting',
        'ai-chat': 'AI Assistant'
    };
    document.getElementById('page-title').textContent = titles[tabName] || 'Dashboard';
    
    // Show/hide tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    
    // Load data if needed
    if (tabName === 'transactions' && state.transactions.length === 0) {
        loadTransactions();
    }
}

// ==================== FILE UPLOAD ====================
function openUploadModal() {
    document.getElementById('upload-modal').classList.add('active');
}

function closeUploadModal() {
    document.getElementById('upload-modal').classList.remove('active');
    state.currentFiles = [];
    document.getElementById('file-list').innerHTML = '';
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
}

function addFiles(files) {
    // Filter PDFs only
    const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    
    if (pdfFiles.length === 0) {
        showToast('Please select PDF files only', 'error');
        return;
    }
    
    if (pdfFiles.length > 10) {
        showToast('Maximum 10 files allowed', 'error');
        return;
    }
    
    state.currentFiles = pdfFiles;
    displayFileList(pdfFiles);
}

function displayFileList(files) {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';
    
    files.forEach((file, idx) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>📄 ${file.name}</span>
            <span style="color: var(--text-muted)">${formatFileSize(file.size)}</span>
        `;
        fileList.appendChild(fileItem);
    });
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function handleUpload() {
    if (state.currentFiles.length === 0) {
        showToast('Please select files first', 'error');
        return;
    }
    
    const formData = new FormData();
    state.currentFiles.forEach(file => {
        formData.append('files', file);
    });
    
    showLoading('Uploading and processing PDFs...');
    
    try {
        // Step 1: Parse PDFs
        const parseRes = await fetch(`${API_BASE}/parse_pdfs`, {
            method: 'POST',
            body: formData
        });
        
        if (!parseRes.ok) throw new Error('PDF parsing failed');
        const parseData = await parseRes.json();
        
        updateLoadingText('Building timeline...');
        
        // Step 2: Build Timeline
        const timelineRes = await fetch(`${API_BASE}/build_timeline`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: parseData.result.output })
        });
        
        if (!timelineRes.ok) throw new Error('Timeline failed');
        const timelineData = await timelineRes.json();
        
        updateLoadingText('Categorizing transactions...');
        
        // Step 3: Categorize
        const categRes = await fetch(`${API_BASE}/categorize_transactions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: timelineData.result.output })
        });
        
        if (!categRes.ok) throw new Error('Categorization failed');
        const categData = await categRes.json();
        
        updateLoadingText('Analyzing patterns...');
        
        // Step 4: Run Analysis & Visualizations in parallel
        await Promise.all([
            fetch(`${API_BASE}/analyze_transactions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categData.result.output })
            }),
            fetch(`${API_BASE}/generate_visualizations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categData.result.output })
            }),
            fetch(`${API_BASE}/generate_stories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_csv: categData.result.output })
            })
        ]);
        
        hideLoading();
        closeUploadModal();
        showToast(`✅ Successfully processed ${state.currentFiles.length} file(s)!`, 'success');
        
        // Reload dashboard
        await loadDashboardData();
        
    } catch (error) {
        console.error('Upload error:', error);
        hideLoading();
        showToast('Upload failed: ' + error.message, 'error');
    }
}

// ==================== DATA LOADING ====================
async function loadInitialData() {
    try {
        await loadDashboardData();
    } catch (error) {
        console.log('No initial data available');
    }
}

async function loadDashboardData() {
    showLoading('Loading dashboard...');
    
    try {
        // Load visualizations
        const vizRes = await fetch(`${API_BASE}/visualizations`);
        if (vizRes.ok) {
            state.visualizations = await vizRes.json();
            updateDashboardMetrics(state.visualizations);
            renderCharts(state.visualizations);
        }
        
        // Load analysis insights
        const analysisRes = await fetch(`${API_BASE}/analyze_transactions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_csv: 'data/output/categorized.csv' })
        });
        
        if (analysisRes.ok) {
            const analysis = await analysisRes.json();
            displayInsights(analysis.result);
        }
        
        hideLoading();
    } catch (error) {
        console.error('Dashboard load error:', error);
        hideLoading();
    }
}

function updateDashboardMetrics(viz) {
    const overview = viz.account_overview;
    
    // Update metric cards
    document.getElementById('total-balance').textContent = `₹${formatNumber(overview.total_balance)}`;
    document.getElementById('monthly-income').textContent = `₹${formatNumber(overview.monthly_income)}`;
    document.getElementById('monthly-expense').textContent = `₹${formatNumber(overview.monthly_expense)}`;
    
    // Update changes
    updateChange('balance-change', overview.balance_percentage);
    updateChange('income-change', overview.income_percentage);
    updateChange('expense-change', overview.expense_percentage);
}

function updateChange(elementId, percentage) {
    const elem = document.getElementById(elementId);
    const sign = percentage >= 0 ? '+' : '';
    elem.textContent = `${sign}${percentage.toFixed(1)}%`;
    elem.className = 'metric-change ' + (percentage >= 0 ? 'positive' : 'negative');
}

function renderCharts(viz) {
    // Spending Trends Chart
    const spendingCtx = document.getElementById('spending-chart').getContext('2d');
    if (spendingChart) spendingChart.destroy();
    
    spendingChart = new Chart(spendingCtx, {
        type: 'line',
        data: {
            labels: viz.spending_trends.labels,
            datasets: [
                {
                    label: 'Expenses',
                    data: viz.spending_trends.expenses,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Budget',
                    data: viz.spending_trends.budget,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    labels: { color: '#f1f5f9' }
                }
            },
            scales: {
                y: { 
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                x: { 
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                }
            }
        }
    });
    
    // Pie Chart
    const pieCtx = document.getElementById('pie-chart').getContext('2d');
    if (pieChart) pieChart.destroy();
    
    pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: viz.expense_breakdown.categories,
            datasets: [{
                data: viz.expense_breakdown.percentages,
                backgroundColor: [
                    '#6366f1', '#8b5cf6', '#ec4899', 
                    '#f59e0b', '#10b981', '#3b82f6'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    position: 'bottom',
                    labels: { color: '#f1f5f9' }
                }
            }
        }
    });
}

function displayInsights(analysis) {
    const container = document.getElementById('insights-container');
    container.innerHTML = '';
    
    // Patterns
    if (analysis.patterns && analysis.patterns.length > 0) {
        analysis.patterns.forEach(pattern => {
            const card = createInsightCard('📊 Pattern', pattern);
            container.appendChild(card);
        });
    }
    
    // Recurring
    if (analysis.recurring && analysis.recurring.length > 0) {
        analysis.recurring.slice(0, 3).forEach(rec => {
            const card = createInsightCard(
                '🔄 Recurring', 
                `${rec.narration}: ₹${rec.amount} (${rec.frequency})`
            );
            container.appendChild(card);
        });
    }
    
    // Anomalies
    if (analysis.anomalies && analysis.anomalies.length > 0) {
        analysis.anomalies.slice(0, 2).forEach(anom => {
            const card = createInsightCard(
                '⚠️ Anomaly',
                `${anom.Narration}: ₹${anom.amount}`,
                anom.severity === 'high' ? 'danger' : 'warning'
            );
            container.appendChild(card);
        });
    }
    
    if (container.children.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted)">No insights available yet. Upload statements to get started.</p>';
    }
}

function createInsightCard(title, content, type = 'info') {
    const card = document.createElement('div');
    card.className = 'insight-card';
    card.innerHTML = `
        <h4>${title}</h4>
        <p>${content}</p>
    `;
    return card;
}

// ==================== TRANSACTIONS ====================
async function loadTransactions() {
    showLoading('Loading transactions...');
    
    try {
        const res = await fetch(`${API_BASE}/transactions`);
        if (!res.ok) throw new Error('No transactions found');
        
        state.transactions = await res.json();
        displayTransactions(state.transactions);
        populateCategoryFilter();
        hideLoading();
    } catch (error) {
        console.error('Transaction load error:', error);
        hideLoading();
        document.getElementById('transactions-list').innerHTML = 
            '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No transactions available. Upload bank statements to get started.</p>';
    }
}

function displayTransactions(transactions) {
    const container = document.getElementById('transactions-list');
    container.innerHTML = '';
    
    transactions.forEach(txn => {
        const amount = txn['Withdrawal (INR)'] || -txn['Deposit (INR)'];
        const isExpense = amount > 0;
        
        const item = document.createElement('div');
        item.className = 'transaction-item';
        item.innerHTML = `
            <div class="transaction-info">
                <div class="transaction-name">${txn.Narration || 'Unknown'}</div>
                <div class="transaction-date">${txn.parsed_date || 'N/A'}</div>
            </div>
            <div class="transaction-amount ${isExpense ? 'negative' : 'positive'}">
                ${isExpense ? '-' : '+'}₹${Math.abs(amount).toFixed(2)}
            </div>
            <div class="category-badge">${txn.category || 'Uncategorized'}</div>
        `;
        container.appendChild(item);
    });
}

function populateCategoryFilter() {
    const select = document.getElementById('filter-category');
    const categories = [...new Set(state.transactions.map(t => t.category))];
    
    categories.forEach(cat => {
        const option = document.createElement('option');
        option.value = cat;
        option.textContent = cat;
        select.appendChild(option);
    });
}

function filterTransactions() {
    const searchTerm = document.getElementById('search-transactions').value.toLowerCase();
    const category = document.getElementById('filter-category').value;
    
    const filtered = state.transactions.filter(txn => {
        const matchesSearch = txn.Narration.toLowerCase().includes(searchTerm);
        const matchesCategory = category === 'all' || txn.category === category;
        return matchesSearch && matchesCategory;
    });
    
    displayTransactions(filtered);
}

// ==================== GRAPH RAG ====================
async function runGraphAnalysis() {
    const btn = document.getElementById('run-graph-analysis');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    showLoading('Running Graph RAG analysis...');
    
    try {
        const res = await fetch(`${API_BASE}/graph-insights`);
        if (!res.ok) throw new Error('Graph analysis failed');
        
        const data = await res.json();
        displayGraphResults(data.insights);
        state.graphData = data.graph_data;
        
        hideLoading();
        document.getElementById('graph-results').classList.remove('hidden');
    } catch (error) {
        console.error('Graph analysis error:', error);
        hideLoading();
        showToast('Graph analysis failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Spending Graph';
    }
}

function displayGraphResults(insights) {
    // Top Merchants
    const merchantsContainer = document.getElementById('top-merchants');
    merchantsContainer.innerHTML = '';
    
    insights.top_merchants.forEach(([merchant, score]) => {
        const item = document.createElement('div');
        item.className = 'merchant-item';
        item.innerHTML = `
            <span class="merchant-name">${merchant}</span>
            <span class="merchant-score">${(score * 100).toFixed(1)}%</span>
        `;
        merchantsContainer.appendChild(item);
    });
    
    // Clusters
    const clustersContainer = document.getElementById('spending-clusters');
    clustersContainer.innerHTML = '';
    
    insights.spending_clusters.forEach(cluster => {
        const item = document.createElement('div');
        item.className = 'cluster-item';
        item.innerHTML = `
            <div>
                <div class="cluster-name">Cluster ${cluster.cluster_id + 1}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted)">
                    ${cluster.categories.join(', ')}
                </div>
            </div>
            <span class="cluster-amount">₹${formatNumber(cluster.total_spent)}</span>
        `;
        clustersContainer.appendChild(item);
    });
    
    // Graph Visualization (simple D3.js visualization)
    renderGraphVisualization(state.graphData);
}

function renderGraphVisualization(graphData) {
    // Basic D3.js graph visualization
    const container = document.getElementById('graph-viz');
    container.innerHTML = '<p style="text-align: center; padding: 2rem; color: var(--text-muted);">Interactive graph visualization would render here with D3.js</p>';
    
    // Placeholder for D3 implementation
    // In production, implement force-directed graph using d3.forceSimulation()
}

// ==================== FORECAST ====================
async function runForecast() {
    const btn = document.getElementById('run-forecast');
    btn.disabled = true;
    btn.textContent = 'Forecasting...';
    showLoading('Generating 30-day forecast...');
    
    try {
        const res = await fetch(`${API_BASE}/forecast?days=30`);
        if (!res.ok) throw new Error('Forecast failed');
        
        const data = await res.json();
        displayForecastResults(data.forecast);
        state.forecastData = data.forecast;
        
        hideLoading();
        document.getElementById('forecast-results').classList.remove('hidden');
    } catch (error) {
        console.error('Forecast error:', error);
        hideLoading();
        showToast('Forecast failed: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Forecast';
    }
}

function displayForecastResults(forecast) {
    // Update summary cards
    document.getElementById('current-balance').textContent = `₹${formatNumber(forecast.current_balance)}`;
    document.getElementById('predicted-balance').textContent = `₹${formatNumber(forecast.predicted_balance_30d)}`;
    
    // Alert card
    const alertCard = document.getElementById('alert-card');
    const alertMsg = document.getElementById('alert-message');
    
    if (forecast.warning_message) {
        alertCard.classList.add('active');
        alertMsg.textContent = forecast.warning_message;
    } else {
        alertCard.classList.remove('active');
        alertMsg.textContent = 'No critical alerts';
    }
    
    // Render forecast chart
    renderForecastChart(forecast.forecast_points);
    
    // Display recommendations
    displayRecommendations(forecast.recommendations);
}

function renderForecastChart(points) {
    const ctx = document.getElementById('forecast-chart').getContext('2d');
    if (forecastChart) forecastChart.destroy();
    
    const dates = points.map(p => p.date);
    const predictions = points.map(p => p.predicted_balance);
    const lowerBounds = points.map(p => p.lower_bound);
    const upperBounds = points.map(p => p.upper_bound);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Predicted Balance',
                    data: predictions,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Lower Bound',
                    data: lowerBounds,
                    borderColor: '#94a3b8',
                    borderDash: [5, 5],
                    fill: false
                },
                {
                    label: 'Upper Bound',
                    data: upperBounds,
                    borderColor: '#94a3b8',
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f1f5f9' } }
            },
            scales: {
                y: { 
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                x: { 
                    ticks: { 
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations-list');
    container.innerHTML = '';
    
    recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.textContent = rec;
        container.appendChild(item);
    });
}

// ==================== AI CHAT ====================
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Add user message
    addChatMessage(query, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const res = await fetch(`${API_BASE}/conversational-query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        const data = await res.json();
        
        removeTypingIndicator(typingId);
        
        if (data.success) {
            addChatMessage(data.response, 'ai');
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage('Sorry, I could not process your request.', 'ai');
    }
}

function addChatMessage(content, type) {
    const container = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = type === 'user' ? 'user-message' : 'ai-message';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    container.appendChild(messageDiv);
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'ai-message typing-indicator';
    typingDiv.id = 'typing-' + Date.now();
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <p>Thinking...</p>
        </div>
    `;
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
    return typingDiv.id;
}

function removeTypingIndicator(id) {
    document.getElementById(id)?.remove();
}

// ==================== UTILITIES ====================
function showLoading(text = 'Loading...') {
    const overlay = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = text;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}

function updateLoadingText(text) {
    document.getElementById('loading-text').textContent = text;
}

function showToast(message, type = 'info') {
    // Simple toast notification (can be enhanced with a library)
    alert(message);
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-IN').format(num);
}

// Export for debugging
window.WealthGuardApp = {
    state,
    loadDashboardData,
    runGraphAnalysis,
    runForecast
};
/* ===== JavaScript文件 - Multi-Agent问答平台 ===== */
/* 这个文件包含了问答平台的所有交互逻辑和功能实现 */
/* 包括消息发送、历史记录管理、UI更新、文件上传等功能 */

/* ===== 全局变量定义 ===== */
/* 用于存储应用程序的全局状态和配置 */
let sessionId = null;                    // 当前会话ID，用于标识用户会话
const API_BASE_URL = 'http://localhost:8000';  // 后端API的基础URL地址

/**
 * 历史对话记录管理类
 * 负责管理用户的历史对话记录，包括保存、获取、删除等功能
 * 使用localStorage进行本地存储，最多保留4次对话记录
 */
class ConversationHistory {
    /**
     * 构造函数 - 初始化历史记录管理器
     * 设置最大历史记录数量、存储键名和当前对话数组
     */
    constructor() {
        this.maxHistory = 4;                 // 最多保留4次对话记录，避免存储过多数据
        this.storageKey = 'chat_history';    // localStorage存储键名
        this.currentConversation = [];       // 当前对话的消息数组
    }
    
    /**
     * 保存当前对话到历史记录
     * 将当前进行中的对话保存到localStorage中
     * 自动生成对话标题和时间戳，并维护最大记录数量限制
     */
    saveCurrentConversation() {
        // 如果当前对话为空，则不保存
        if (this.currentConversation.length === 0) return;
        
        // 创建对话记录对象
        const conversation = {
            id: Date.now(),                      // 使用时间戳作为唯一ID
            timestamp: new Date().toISOString(), // ISO格式的时间戳
            messages: [...this.currentConversation], // 复制当前对话消息数组
            title: this.generateConversationTitle()  // 自动生成对话标题
        };
        
        // 获取现有历史记录
        let history = this.getHistory();
        // 将新对话添加到数组开头（最新的在前面）
        history.unshift(conversation);
        
        // 保持最多4次记录，删除超出的旧记录
        if (history.length > this.maxHistory) {
            history = history.slice(0, this.maxHistory);
        }
        
        // 保存到localStorage
        localStorage.setItem(this.storageKey, JSON.stringify(history));
        console.log('对话已保存到历史记录');  // 调试信息
    }
    
    /**
     * 获取历史记录
     * 从localStorage中读取并解析历史对话记录
     * @returns {Array} 历史对话记录数组，如果没有记录则返回空数组
     */
    getHistory() {
        const stored = localStorage.getItem(this.storageKey);
        return stored ? JSON.parse(stored) : [];  // 解析JSON或返回空数组
    }
    
    /**
     * 添加消息到当前对话
     * 将用户消息或AI回复添加到当前对话记录中
     * @param {string} type - 消息类型（'user' 或 'ai'）
     * @param {string} content - 消息内容
     */
    addMessage(type, content) {
        // 创建消息对象并添加到当前对话数组
        this.currentConversation.push({
            type: type,                          // 消息类型：'user'（用户）或'ai'（AI回复）
            content: content,                    // 消息内容文本
            timestamp: new Date().toISOString()  // 消息时间戳，ISO格式
        });
    }
    
    /**
     * 清空当前对话
     * 重置当前对话数组，用于开始新的对话
     */
    clearCurrentConversation() {
        this.currentConversation = [];  // 清空当前对话消息数组
    }
    
    /**
     * 生成对话标题
     * 自动生成对话标题，取第一条用户消息的前20个字符作为标题
     * 如果超过20个字符则添加省略号
     * @returns {string} 生成的对话标题
     */
    generateConversationTitle() {
        // 查找第一条用户消息
        const firstUserMessage = this.currentConversation.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            // 截取前20个字符作为标题
            const title = firstUserMessage.content.substring(0, 20);
            // 如果原文本更长，则添加省略号
            return title.length < firstUserMessage.content.length ? title + '...' : title;
        }
        return '新对话';  // 如果没有用户消息，返回默认标题
    }
    
    /**
     * 加载指定的历史对话
     * 根据对话ID从历史记录中加载指定对话，并在界面上重新显示
     * @param {number} conversationId - 要加载的对话ID
     */
    loadConversation(conversationId) {
        const history = this.getHistory();  // 获取所有历史记录
        // 根据ID查找指定对话
        const conversation = history.find(conv => conv.id === conversationId);
        if (conversation) {
            // 清空当前界面的对话容器
            const container = document.getElementById('qaContainer');
            container.innerHTML = '';
            
            // 重新显示历史对话中的所有消息
            conversation.messages.forEach(msg => {
                // false参数表示不将这些消息添加到当前对话记录中（避免重复）
                addMessageToUI(msg.type, msg.content, false);
            });
            
            // 将加载的对话设置为当前对话（复制数组避免引用问题）
            this.currentConversation = [...conversation.messages];
        }
    }
    
    /**
     * 删除指定的历史对话
     * 从localStorage中删除指定ID的历史对话记录
     * @param {number} conversationId - 要删除的对话ID
     */
    deleteConversation(conversationId) {
        let history = this.getHistory();  // 获取当前历史记录
        // 过滤掉指定ID的对话，保留其他对话
        history = history.filter(conv => conv.id !== conversationId);
        // 更新localStorage中的历史记录
        localStorage.setItem(this.storageKey, JSON.stringify(history));
    }
}

/* ===== 全局实例创建 ===== */
// 创建历史记录管理实例，用于整个应用程序的对话历史管理
const conversationHistory = new ConversationHistory();

/* ===== 消息发送和处理函数 ===== */

/**
 * 发送消息到后端API
 * 主要的消息发送函数，处理用户输入并调用后端API
 * 包括输入验证、UI更新、错误处理等功能
 */
async function sendMessage() {
    // 获取输入框元素和用户输入的消息
    const input = document.getElementById('messageInput');
    const message = input.value.trim();  // 去除首尾空格
    
    // 如果消息为空，则不发送
    if (!message) return;
    
    // 清空输入框，为下次输入做准备
    input.value = '';
    
    // 立即在界面上显示用户消息，提供即时反馈
    addMessageToUI('user', message);
    
    try {
        // 调用SSE流式API获取AI响应
        await streamChatResponse(message);
    } catch (error) {
        // 错误处理：记录错误并向用户显示友好的错误信息
        console.error('发送消息失败:', error);
        addMessageToUI('error', '抱歉，发送消息时出现错误，请稍后重试。');
    }
}

/**
 * 使用SSE流式接收AI响应
 * 通过Server-Sent Events (SSE) 技术实现流式响应
 * 允许AI回复以流的形式逐步显示，提供更好的用户体验
 * @param {string} message - 用户发送的消息内容
 */
async function streamChatResponse(message) {
    // 构建请求体，包含用户消息
    const requestBody = {
        message: message  // 用户输入的消息内容
    };
    
    // 如果存在会话ID，则添加到请求体中（用于维持对话上下文）
    if (sessionId) {
        requestBody.session_id = sessionId;
    }
    
    // 发送POST请求到流式聊天API端点
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',                          // 使用POST方法
        headers: {
            'Content-Type': 'application/json'   // 设置请求内容类型为JSON
        },
        body: JSON.stringify(requestBody)        // 将请求体转换为JSON字符串
    });
    
    // 检查HTTP响应状态，如果不成功则抛出错误
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    // 获取响应流的读取器和文本解码器
    const reader = response.body.getReader();  // 用于读取流式数据
    const decoder = new TextDecoder();         // 用于将字节流解码为文本
    let aiMessageElement = null;  // 用于存储AI消息的DOM元素引用
    
    try {
        // 持续读取流式数据直到完成
        while (true) {
            // 从流中读取数据块
            const { done, value } = await reader.read();
            if (done) break;  // 如果读取完成，退出循环
            
            // 将字节数据解码为文本
            const chunk = decoder.decode(value);
            // 按行分割数据（SSE格式是按行传输的）
            const lines = chunk.split('\n');
            
            // 处理每一行数据
            for (const line of lines) {
                // 检查是否是SSE数据行（以'data: '开头）
                if (line.startsWith('data: ')) {
                    // 提取实际的数据内容（去掉'data: '前缀）
                    const data = line.slice(6);
                    
                    // 检查是否是结束标志
                    if (data === '[DONE]') {
                        return;  // 流式传输完成，退出函数
                    }
                    
                    try {
                        // 解析JSON数据
                        const parsed = JSON.parse(data);
                        
                        // 根据消息类型进行不同的处理
                        switch (parsed.type) {
                            case 'session_id':
                                // 接收并保存会话ID，用于后续请求
                                sessionId = parsed.session_id;
                                break;
                            case 'ai_start':
                                // AI开始回复，创建空的AI消息元素
                                aiMessageElement = addMessageToUI('ai', '');
                                break;
                            case 'ai_chunk':
                                // 接收AI回复的文本片段，逐步追加到消息元素
                                if (aiMessageElement) {
                                    aiMessageElement.textContent += parsed.content;
                                }
                                break;
                            case 'message':
                                // 接收完整的AI消息（用于非流式模式）
                                if (!aiMessageElement) {
                                    aiMessageElement = addMessageToUI('ai', parsed.content);
                                } else {
                                    aiMessageElement.textContent = parsed.content;
                                }
                                break;
                            case 'error':
                                // 接收错误消息并显示给用户
                                addMessageToUI('error', parsed.content);
                                break;
                        }
                    } catch (parseError) {
                        // JSON解析错误处理
                        console.error('解析SSE数据失败:', parseError);
                    }
                }
            }
        }
    } finally {
        // 确保释放流读取器的锁定，避免资源泄漏
        reader.releaseLock();
    }
}

/* ===== UI界面更新函数 ===== */

/**
 * 添加消息到UI界面（聊天气泡样式）
 * 这是核心的UI更新函数，负责在界面上显示用户消息、AI回复和错误信息
 * 支持不同类型的消息样式和自动滚动到最新消息
 * @param {string} type - 消息类型 ('user', 'ai', 'error')
 * @param {string} content - 消息内容
 * @param {boolean} addToHistory - 是否添加到历史记录，默认为true
 * @returns {HTMLElement} 返回创建的消息元素（AI消息返回内容元素，其他返回整个消息容器）
 */
function addMessageToUI(type, content, addToHistory = true) {
    // 获取消息容器元素
    const container = document.getElementById('qaContainer');
    // 创建消息容器div
    const messageDiv = document.createElement('div');
    // 设置CSS类名，包含通用类和类型特定类
    messageDiv.className = `chat-message ${type}`;
    
    // 添加到历史记录（如果需要且是有效的消息类型）
    if (addToHistory && (type === 'user' || type === 'ai')) {
        conversationHistory.addMessage(type, content);
    }
    
    // 根据消息类型创建不同的UI结构
    if (type === 'user') {
        // 用户消息：右侧对齐，显示用户名和头像
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-info">chunkit</div>
                <div class="message-bubble">${formatMessageContent(content)}</div>
            </div>
            <div class="message-avatar user"></div>
        `;
    } else if (type === 'ai') {
        // AI消息：左侧对齐，显示AI标识和助手名称
        // 创建独立的答案div，用于流式更新内容
        const answerDiv = document.createElement('div');
        answerDiv.className = 'message-bubble';
        answerDiv.innerHTML = formatMessageContent(content);
        
        // 构建AI消息的基本结构
        messageDiv.innerHTML = `
            <div class="message-avatar ai">AI</div>
            <div class="message-content">
                <div class="message-info">figma助手</div>
            </div>
        `;
        
        // 将答案div添加到消息内容中
        messageDiv.querySelector('.message-content').appendChild(answerDiv);
        // 立即添加到容器并滚动到视图（用于流式显示）
        container.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
        
        // 返回答案div，用于后续的流式内容更新
        return answerDiv;
    } else if (type === 'error') {
        // 错误消息：特殊样式，红色主题
        messageDiv.innerHTML = `
            <div class="message-avatar ai">⚠️</div>
            <div class="message-content">
                <div class="message-info">系统提示</div>
                <div class="message-bubble" style="background: #ffebee; color: #c62828; border-color: #ef5350;">${content}</div>
            </div>
        `;
    }
    
    // 将消息添加到容器中
    container.appendChild(messageDiv);
    // 平滑滚动到最新消息，确保用户能看到最新内容
    messageDiv.scrollIntoView({ behavior: 'smooth' });
    
    // 返回整个消息div元素
    return messageDiv;
}

/* ===== 消息内容格式化函数 ===== */

/**
 * 格式化消息内容，处理文件引用和关键词高亮
 * 使用正则表达式识别特定模式并添加样式，提升消息的可读性
 * @param {string} content - 原始消息内容
 * @returns {string} 格式化后的HTML内容
 */
function formatMessageContent(content) {
    // 处理文件引用格式 (例如: styles.css 105-120)
    // 匹配"文件名.扩展名 数字-数字"的模式，添加文件图标和特殊样式
    content = content.replace(
        /(\w+\.\w+)\s+(\d+-\d+)/g,
        '<span class="file-reference"><span class="file-icon">📄</span>$1 $2</span>'
    );
    
    // 处理关键词高亮格式 (例如: # styles.css)
    // 匹配"# 文件名.扩展名"的模式，添加高亮样式
    content = content.replace(
        /#\s*(\w+\.\w+)/g,
        '<span class="keyword-highlight"># $1</span>'
    );
    
    return content;  // 返回处理后的HTML内容
}

/* ===== 对话管理函数 ===== */

/**
 * 开始新对话
 * 完整的新对话流程，包括保存当前对话、清理状态、重置界面
 * 提供流畅的用户体验和完整的数据管理
 */
function startNewConversation() {
    // 1. 保存当前对话到历史记录（如果有内容的话）
    conversationHistory.saveCurrentConversation();
    
    // 2. 如果存在当前会话，先清空后端历史记录
    if (sessionId) {
        clearChatHistory();
    }
    
    // 3. 重置会话状态，准备新的会话
    sessionId = null;
    
    // 4. 清空当前对话记录数组
    conversationHistory.clearCurrentConversation();
    
    // 5. 清空聊天容器中的所有消息DOM元素
    const container = document.getElementById('qaContainer');
    container.innerHTML = '';
    
    // 6. 清空并聚焦输入框，方便用户开始新对话
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.value = '';        // 清空输入框内容
        messageInput.focus();           // 聚焦到输入框，提升用户体验
    }
    
    // 7. 显示欢迎消息（不添加到历史记录，避免污染对话数据）
    addMessageToUI('ai', '您好！我是Multi-Agent问答助手，有什么可以帮助您的吗？', false);
    
    console.log('新对话已开始');  // 调试信息
}

/**
 * 清空聊天历史
 * 向后端API发送请求，清空指定会话的历史记录
 * 用于释放服务器资源和重置对话上下文
 */
async function clearChatHistory() {
    // 如果没有会话ID，则无需清空
    if (!sessionId) return;
    
    try {
        // 发送DELETE请求到后端API
        const response = await fetch(`${API_BASE_URL}/api/chat/history/${sessionId}`, {
            method: 'DELETE'  // 使用DELETE方法清空历史记录
        });
        
        // 检查响应状态
        if (response.ok) {
            console.log('聊天历史已清空');  // 成功日志
        } else {
            console.error('清空聊天历史失败');  // 失败日志
        }
    } catch (error) {
        // 网络错误或其他异常处理
        console.error('清空聊天历史时出错:', error);
    }
}

/* ===== 历史记录管理函数 ===== */

/**
 * 显示历史记录
 * 创建模态弹窗显示所有历史对话记录
 * 提供加载和删除历史对话的交互功能
 */
function showHistory() {
    // 获取所有历史对话记录
    const history = conversationHistory.getHistory();
    
    // 如果没有历史记录，显示提示信息
    if (history.length === 0) {
        alert('暂无历史对话记录');
        return;
    }
    
    // 创建历史记录模态弹窗DOM元素
    const modal = document.createElement('div');
    modal.className = 'history-modal';
    // 使用模板字符串构建弹窗HTML结构
    modal.innerHTML = `
        <div class="history-modal-content">
            <div class="history-header">
                <h3>历史对话记录</h3>
                <button class="close-btn" onclick="closeHistoryModal()">&times;</button>
            </div>
            <div class="history-list">
                ${history.map(conv => `
                    <div class="history-item" data-id="${conv.id}">
                        <div class="history-item-content">
                            <div class="history-title">${conv.title}</div>
                            <div class="history-time">${formatTime(conv.timestamp)}</div>
                            <div class="history-preview">${conv.messages.length} 条消息</div>
                        </div>
                        <div class="history-actions">
                            <button class="load-btn" onclick="loadHistoryConversation(${conv.id})">加载</button>
                            <button class="delete-btn" onclick="deleteHistoryConversation(${conv.id})">删除</button>
                        </div>
                    </div>
                `).join('')}  <!-- 将历史记录数组转换为HTML字符串 -->
            </div>
        </div>
        <div class="history-modal-overlay" onclick="closeHistoryModal()"></div>
    `;
    
    // 将模态弹窗添加到页面body中
    document.body.appendChild(modal);
}

/**
 * 关闭历史记录弹窗
 * 从DOM中移除历史记录模态弹窗元素
 */
function closeHistoryModal() {
    // 查找并移除历史记录模态弹窗
    const modal = document.querySelector('.history-modal');
    if (modal) {
        modal.remove();  // 从DOM中移除元素
    }
}

/**
 * 加载历史对话
 * 将指定的历史对话加载到当前界面，替换当前对话内容
 * @param {number} conversationId - 要加载的历史对话ID
 */
function loadHistoryConversation(conversationId) {
    // 先保存当前对话到历史记录（避免丢失当前对话）
    conversationHistory.saveCurrentConversation();
    
    // 加载指定的历史对话到界面
    conversationHistory.loadConversation(conversationId);
    
    // 关闭历史记录弹窗
    closeHistoryModal();
    
    console.log('历史对话已加载');  // 调试信息
}

/**
 * 删除历史对话
 * 删除指定的历史对话记录，需要用户确认
 * @param {number} conversationId - 要删除的历史对话ID
 */
function deleteHistoryConversation(conversationId) {
    // 显示确认对话框，防止误删除
    if (confirm('确定要删除这条历史对话吗？')) {
        // 从localStorage中删除指定的历史对话
        conversationHistory.deleteConversation(conversationId);
        
        // 刷新历史记录显示（关闭当前弹窗并重新打开）
        closeHistoryModal();
        showHistory();
        
        console.log('历史对话已删除');  // 调试信息
    }
}

/* ===== 工具函数 ===== */

/**
 * 格式化时间显示
 * 将时间戳转换为用户友好的相对时间格式
 * 支持"刚刚"、"X分钟前"、"X小时前"和具体日期时间
 * @param {string} timestamp - ISO格式的时间戳
 * @returns {string} 格式化后的时间字符串
 */
function formatTime(timestamp) {
    const date = new Date(timestamp);  // 将时间戳转换为Date对象
    const now = new Date();            // 获取当前时间
    const diff = now - date;           // 计算时间差（毫秒）
    
    // 根据时间差返回不同的格式
    if (diff < 60000) { // 1分钟内（60000毫秒）
        return '刚刚';
    } else if (diff < 3600000) { // 1小时内（3600000毫秒）
        return `${Math.floor(diff / 60000)}分钟前`;
    } else if (diff < 86400000) { // 24小时内（86400000毫秒）
        return `${Math.floor(diff / 3600000)}小时前`;
    } else {
        // 超过24小时，显示具体的日期和时间
        return date.toLocaleDateString('zh-CN', {
            month: 'short',    // 短月份格式
            day: 'numeric',    // 数字日期
            hour: '2-digit',   // 两位数小时
            minute: '2-digit'  // 两位数分钟
        });
    }
}

/* ===== 系统功能函数 ===== */

/**
 * 健康检查
 * 检查后端API服务是否正常运行
 * 用于确保前后端连接正常
 */
async function healthCheck() {
    try {
        // 向后端健康检查端点发送请求
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            console.log('后端服务正常');  // 服务正常日志
        } else {
            console.warn('后端服务异常');  // 服务异常警告
        }
    } catch (error) {
        // 网络错误或服务不可达
        console.error('健康检查失败:', error);
    }
}

/**
 * 页面初始化函数
 * 设置页面的各种事件监听器和初始状态
 * 在页面加载完成后调用
 */
function initializePage() {
    // 进行后端服务健康检查
    healthCheck();
    
    // 绑定回车键发送消息功能
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            // 检查是否按下回车键且没有按住Shift键
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();  // 阻止默认的换行行为
                sendMessage();       // 发送消息
            }
        });
    }
    
    // 页面卸载时自动保存当前对话
    window.addEventListener('beforeunload', function() {
        conversationHistory.saveCurrentConversation();
    });
    
    console.log('页面初始化完成，历史记录功能已启用');
}

/* ===== 文件上传功能 ===== */

/**
 * 触发文件上传选择
 * 程序化地触发隐藏的文件输入框，打开文件选择对话框
 */
function triggerFileUpload() {
    const fileInput = document.getElementById('fileInput');
    fileInput.click();  // 模拟点击文件输入框
}

/**
 * 处理文件上传
 * 处理用户选择的文件，上传到后端并获取分析结果
 * @param {File} file - 用户选择的文件对象
 */
async function handleFileUpload(file) {
    // 如果没有文件，直接返回
    if (!file) return;
    
    // 显示文件上传状态信息
    const fileName = file.name;
    const fileSize = (file.size / 1024 / 1024).toFixed(2); // 转换为MB并保留2位小数
    addMessageToUI('user', `上传文件: ${fileName} (${fileSize}MB)`);
    
    try {
        // 创建FormData对象用于文件上传
        const formData = new FormData();
        formData.append('file', file);  // 添加文件
        formData.append('user_message', `请分析这个文件: ${fileName}`);  // 添加用户消息
        
        // 如果存在会话ID，添加到请求中
        if (sessionId) {
            formData.append('session_id', sessionId);
        }
        
        // 调用文件上传API，使用流式响应处理
        await streamFileUploadResponse(formData);
    } catch (error) {
        // 文件上传错误处理
        console.error('文件上传失败:', error);
        addMessageToUI('error', '抱歉，文件上传时出现错误，请稍后重试。');
    }
}

/**
 * 使用SSE流式接收文件上传响应
 * 处理文件上传后的AI分析结果，支持流式显示
 * 与普通消息的流式处理逻辑相同，但使用不同的API端点
 * @param {FormData} formData - 包含文件和相关信息的表单数据
 */
async function streamFileUploadResponse(formData) {
    // 向文件处理API端点发送POST请求
    const response = await fetch(`${API_BASE_URL}/api/files/chat`, {
        method: 'POST',
        body: formData  // 直接发送FormData，浏览器会自动设置正确的Content-Type
    });
    
    // 检查HTTP响应状态
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    // 设置流式数据读取器和解码器
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let aiMessageElement = null;  // 用于存储AI回复的DOM元素
    
    try {
        // 持续读取流式响应数据
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;  // 流结束，退出循环
            
            // 解码字节数据为文本
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            // 处理每一行SSE数据
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);  // 提取数据内容
                    
                    // 检查流结束标志
                    if (data === '[DONE]') {
                        return;
                    }
                    
                    try {
                        // 解析JSON数据
                        const parsed = JSON.parse(data);
                        
                        // 根据消息类型处理不同的响应
                        switch (parsed.type) {
                            case 'session_id':
                                // 保存会话ID
                                sessionId = parsed.session_id;
                                break;
                            case 'ai_start':
                                // AI开始回复，创建消息元素
                                aiMessageElement = addMessageToUI('ai', '');
                                break;
                            case 'ai_chunk':
                                // 接收AI回复片段，逐步更新内容
                                if (aiMessageElement) {
                                    aiMessageElement.textContent += parsed.content;
                                }
                                break;
                            case 'message':
                                // 接收完整消息（兼容模式）
                                if (!aiMessageElement) {
                                    aiMessageElement = addMessageToUI('ai', parsed.content);
                                } else {
                                    aiMessageElement.textContent = parsed.content;
                                }
                                break;
                            case 'error':
                                // 显示错误消息
                                addMessageToUI('error', parsed.content);
                                break;
                        }
                    } catch (parseError) {
                        // JSON解析错误处理
                        console.error('解析SSE数据失败:', parseError);
                    }
                }
            }
        }
    } finally {
        // 确保释放流读取器的锁定，避免资源泄漏
        reader.releaseLock();
    }
}

/* ===== 页面加载和事件绑定 ===== */

// 页面加载完成后自动初始化
// 使用DOMContentLoaded事件确保DOM元素完全加载后再执行初始化
document.addEventListener('DOMContentLoaded', function() {
    // 调用页面初始化函数
    initializePage();
    
    // 绑定文件选择事件监听器
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            // 获取用户选择的第一个文件
            const file = e.target.files[0];
            if (file) {
                // 处理文件上传
                handleFileUpload(file);
                // 清空文件输入值，允许用户重复选择同一文件
                e.target.value = '';
            }
        });
    }
});

/* ===== 文件结束 ===== */
/* 这个JavaScript文件包含了Multi-Agent问答平台的完整前端逻辑 */
/* 主要功能包括：消息发送接收、历史记录管理、文件上传、UI交互等 */
/* 使用了现代JavaScript特性：ES6+语法、async/await、流式API等 */
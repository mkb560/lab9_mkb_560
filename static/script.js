document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('pdf-upload');
    const statusDiv = document.getElementById('upload-status');
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');

    // Handle Auto-Upload on File Selection
    fileInput.addEventListener('change', async () => {
        const files = fileInput.files;
        
        // If the user opened the file menu but clicked 'Cancel', do nothing
        if (files.length === 0) return; 

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        statusDiv.textContent = 'Uploaded';
        statusDiv.className = 'status-message analyzing';
        fileInput.disabled = true; // Prevent selecting more files during upload

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            if (response.ok) {
                statusDiv.textContent = 'Analysis complete!';
                statusDiv.className = 'status-message success';
            } else {
                statusDiv.textContent = result.error || 'Upload failed.';
                statusDiv.className = 'status-message error';
            }
        } catch (error) {
            statusDiv.textContent = 'Network error during upload.';
            statusDiv.className = 'status-message error';
        } finally {
            fileInput.disabled = false; 
            fileInput.value = ''; 
        }
    });

    // Helper to add messages to the Chat Window
    const appendMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.classList.add('bubble');
        bubbleDiv.textContent = text;
        
        msgDiv.appendChild(bubbleDiv);
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    // Handle User Asking a Question
    const askQuestion = async () => {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        //userInput.value = '';
        
        const loadingId = 'loading-' + Date.now();
        const tempMsgDiv = document.createElement('div');
        tempMsgDiv.classList.add('message', 'bot-message');
        tempMsgDiv.id = loadingId;
        tempMsgDiv.innerHTML = '<div class="bubble" style="color:#888; font-style:italic;">Thinking...</div>';
        chatWindow.appendChild(tempMsgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text })
            });
            const result = await response.json();
            
            document.getElementById(loadingId).remove();

            if (response.ok) {
                appendMessage(result.answer, 'bot');
            } else {
                appendMessage('Error: ' + result.error, 'bot');
            }
        } catch (error) {
            document.getElementById(loadingId).remove();
            appendMessage('Network error while fetching response.', 'bot');
        }
    };

    sendBtn.addEventListener('click', askQuestion);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') askQuestion();
    });
});
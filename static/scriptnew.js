class ChatPage {
    constructor() {
        this.args = {
            chatInput: document.querySelector("#chat-input"),
            sendButton: document.querySelector("#send-btn"),
            chatContainer: document.querySelector(".chat-container"),
            themeButton: document.querySelector("#theme-btn"),
            deleteButton: document.querySelector("#delete-btn"),
        }

        this.messages= []
    }

    userSend() {
        const {chatInput, sendButton} = this.args; // this.args(chatInput, sendButton);
        const initialInputHeight = chatInput.scrollHeight;

        sendButton.addEventListener("click", () => this.handleOutgoingChat(chatInput))

        const nodeChatInput = chatInput.querySelector('input');
        nodeChatInput.addEventListener("input", () => {
            chatInput.style.height =  `${initialInputHeight}px`;
            chatInput.style.height = `${chatInput.scrollHeight}px`;
        })
        nodeChatInput.addEventListener("keydown", (e) => {
            // If the Enter key is pressed without Shift and the window width is larger 
            // than 800 pixels, handle the outgoing chat
            if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
                e.preventDefault();
                handleOutgoingChat(chatInput);
            }
        })
    }

    handleOutgoingChat(chatInput) {
        var textField = chatInput.querySelector("input");
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = {name: "User", message: text1}
        this.messages.push(msg1)      

        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg2 = { name: "ORCA", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatInput);
            textFieldReformat();
            
        }).catch((error) => {
            console.error("Error: ", error);
            this.updateChatText(chatInput);
            textFieldReformat();
        })
    }

    textFieldReformat() {
        const {chatContainer} = this.args.chatContainer;
        textField.value = "";
        chatInput.style.height = `${initialInputHeight}px`;

        const html = `<div class="chat-content">
                        <div class="chat-details">
                            <img src="images/user.jpg" alt="user-img">
                            <p>${textField}</p>
                        </div>
                    </div>`;

        const nodeChatContainer = chatContainer.querySelector(".default-text");
        nodeChatContainer?.remove();
        nodeChatContainer.appendChild(outgoingChatDiv);
        nodeChatContainer.scrollTo(0, chatContainer.scrollHeight);
        setTimeout(showTypingAnimation, 500);
    }

    showTypingAnimation() {
        const html = `<div class="chat-content">
                        <div class="chat-details">
                            <img src="images/chatbot.jpg" alt="chatbot-img">
                            <div class="typing-animation">
                                <div class="typing-dot" style="--delay: 0.2s"></div>
                                <div class="typing-dot" style="--delay: 0.3s"></div>
                                <div class="typing-dot" style="--delay: 0.4s"></div>
                            </div>
                        </div>
                        <span onclick="copyResponse(this)" class="material-symbols-rounded">content_copy</span>
                    </div>`;
        // Create an incoming chat div with typing animation and append it to chat container
        const incomingChatDiv = createChatElement(html, "incoming");
        chatContainer.appendChild(incomingChatDiv);
        chatContainer.scrollTo(0, chatContainer.scrollHeight);
        getChatResponse(incomingChatDiv);
    }
}

userSend();
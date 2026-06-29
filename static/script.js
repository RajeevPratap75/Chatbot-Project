const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messageList = document.querySelector("#messageList");
const sendButton = document.querySelector("#sendButton");
const statusPill = document.querySelector("#statusPill");
let socket;
let pendingMessage;

function setStatus(label, state = "ready") {
    statusPill.className = `status-pill ${state === "ready" ? "" : state}`.trim();
    statusPill.lastChild.textContent = ` ${label}`;
}

function addMessage(role, text, options = {}) {
    const message = document.createElement("article");
    message.className = `message ${role}${options.typing ? " typing" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "Y" : "A";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = DOMPurify.sanitize(
    marked.parse(text)
);

    message.append(avatar, bubble);
    messageList.append(message);
    messageList.scrollTop = messageList.scrollHeight;

    return message;
}

function getBubble(message) {
    return message.querySelector(".bubble");
}

function autoresize() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
}

async function sendMessage(message) {
    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
    });

    if (!response.ok) {
        let detail = "The server could not answer right now.";
        try {
            const data = await response.json();
            detail = data.detail || detail;
        } catch {
            detail = response.statusText || detail;
        }
        throw new Error(detail);
    }

    return response.json();
}

function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws`;
}

function connectWebSocket() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        return Promise.resolve(socket);
    }

    return new Promise((resolve, reject) => {
        socket = new WebSocket(getWebSocketUrl());

        socket.addEventListener("open", () => resolve(socket), { once: true });
        socket.addEventListener("error", () => reject(new Error("WebSocket connection failed.")), { once: true });

        socket.addEventListener("message", (event) => {
            if (!pendingMessage) {
                return;
            }

            let data;
            try {
                data = JSON.parse(event.data);
            } catch {
                return;
            }

            const bubble = getBubble(pendingMessage.element);

            if (data.type === "status") {
                if (data.message === "retrieval_complete") {
                    setStatus("Generating", "busy");
                    if (!pendingMessage.hasContent) {
                        bubble.textContent = "";
                    }
                } else if (data.message === "streaming_unavailable") {
                    setStatus("Finishing", "busy");
                }
                return;
            }

            if (data.type === "token") {
                if (!pendingMessage.hasContent) {
                    pendingMessage.markdown = "";
                    pendingMessage.element.classList.remove("typing");
                    pendingMessage.hasContent = true;
                }
                
                pendingMessage.markdown += data.content;
                bubble.innerHTML = DOMPurify.sanitize(
                    marked.parse(pendingMessage.markdown)
                );
                
                messageList.scrollTop = messageList.scrollHeight;
                return;
            }

            if (data.type === "final") {
                bubble.innerHTML = marked.parse(
                    data.content || "I did not receive a reply from the server."
                );
                pendingMessage.element.classList.remove("typing");
                pendingMessage.hasContent = true;
                messageList.scrollTop = messageList.scrollHeight;
                return;
            }

            if (data.type === "error") {
                pendingMessage.reject(new Error(data.message || "The server could not answer right now."));
                pendingMessage = null;
                return;
            }

            if (data.type === "done") {
                pendingMessage.resolve();
                pendingMessage = null;
            }
        });

        socket.addEventListener("close", () => {
            if (pendingMessage) {
                pendingMessage.reject(new Error("WebSocket disconnected."));
                pendingMessage = null;
            }
        });
    });
}

async function sendMessageOverWebSocket(message, assistantMessage) {
    const ws = await connectWebSocket();

    return new Promise((resolve, reject) => {
        pendingMessage = {
            element: assistantMessage,
            hasContent: false,
            markdown: "",
            resolve,
            reject
        };
        ws.send(JSON.stringify({ message }));
    });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message) {
        return;
    }

    addMessage("user", message);
    input.value = "";
    autoresize();
    input.focus();

    sendButton.disabled = true;
    setStatus("Thinking", "busy");
    const typingMessage = addMessage("assistant", "Reading the uploaded documents...", { typing: true });

    try {
        try {
            await sendMessageOverWebSocket(message, typingMessage);
        } catch {
            const data = await sendMessage(message);
            getBubble(typingMessage).innerHTML = marked.parse(
                data.reply || "I did not receive a reply from the server."
            );
            typingMessage.classList.remove("typing");
        }
        setStatus("Ready");
    } catch (error) {
        typingMessage.remove();
        addMessage("assistant", error.message);
        setStatus("Error", "error");
    } finally {
        sendButton.disabled = false;
    }
});

input.addEventListener("input", autoresize);

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

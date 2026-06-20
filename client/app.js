const API = window.location.origin;
const SESSION_KEY = "docagent_session";

const $ = (id) => document.getElementById(id);

const chatScroll = $("chatScroll");
const messagesEl = $("messages");
const emptyState = $("emptyState");
const input = $("input");
const sendBtn = $("sendBtn");
const fileInput = $("fileInput");
const filePreview = $("filePreview");
const fileNameEl = $("fileName");

let sessionId = localStorage.getItem(SESSION_KEY) || crypto.randomUUID();
localStorage.setItem(SESSION_KEY, sessionId);

let pendingFile = null;
let isSending = false;
let typingEl = null;

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function updateEmpty() {
  const hasMessages = messagesEl.children.length > 0;
  emptyState.classList.toggle("hidden", hasMessages);
}

function scrollBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function setFile(file) {
  pendingFile = file;
  if (file) {
    fileNameEl.textContent = file.name;
    filePreview.classList.remove("hidden");
  } else {
    filePreview.classList.add("hidden");
    fileNameEl.textContent = "";
    fileInput.value = "";
  }
}

function canSend() {
  return input.value.trim().length > 0 && !isSending;
}

function updateSendBtn() {
  sendBtn.disabled = !canSend();
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

function addMessage(role, text, opts = {}) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}${opts.error ? " error" : ""}`;

  const avatar = role === "user" ? "你" : "AI";
  let fileHtml = "";
  if (opts.file) {
    fileHtml = `<div class="msg-file">📎 ${esc(opts.file)}</div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-text">${esc(text)}</div>
      ${fileHtml}
    </div>
  `;

  messagesEl.appendChild(row);
  updateEmpty();
  scrollBottom();
  return row;
}

function showTyping() {
  removeTyping();
  typingEl = document.createElement("div");
  typingEl.className = "msg-row assistant typing";
  typingEl.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <div class="msg-text typing-dots">思考中<span>.</span><span>.</span><span>.</span></div>
    </div>
  `;
  messagesEl.appendChild(typingEl);
  updateEmpty();
  scrollBottom();
}

function removeTyping() {
  if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

async function checkHealth() {
  const dot = $("statusDot");
  const text = $("statusText");
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const ok = data.status === "healthy" && data.agent === "ready";
    dot.className = "status-dot " + (ok ? "online" : "");
    text.textContent = ok ? "在线" : "初始化中";
  } catch {
    dot.className = "status-dot offline";
    text.textContent = "离线";
  }
}

async function loadHistory() {
  try {
    const res = await fetch(`${API}/chat/history?session_id=${encodeURIComponent(sessionId)}`);
    const data = await res.json();
    if (!data.messages?.length) return;

    messagesEl.innerHTML = "";
    data.messages.forEach((msg) => {
      const meta = msg.metadata || {};
      const file = meta.doc_path ? meta.doc_path.split(/[/\\]/).pop() : null;
      addMessage(msg.role, msg.content, { file: msg.role === "user" ? file : null });
    });
  } catch {
    /* ignore */
  }
}

async function send() {
  const text = input.value.trim();
  if (!text || isSending) return;

  const file = pendingFile;
  addMessage("user", text, { file: file?.name });

  input.value = "";
  autoGrow();
  setFile(null);
  isSending = true;
  updateSendBtn();
  input.disabled = true;
  showTyping();

  const form = new FormData();
  form.append("message", text);
  form.append("session_id", sessionId);
  if (file) form.append("file", file);

  try {
    const res = await fetch(`${API}/chat`, { method: "POST", body: form });
    const data = await res.json();

    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
    }

    removeTyping();
    const reply = data.reply || data.final_result || data.error || "无响应";
    addMessage("assistant", reply, { error: !data.success });
  } catch (e) {
    removeTyping();
    addMessage("assistant", e.message || "请求失败，请检查服务是否启动", { error: true });
  } finally {
    isSending = false;
    input.disabled = false;
    updateSendBtn();
    input.focus();
  }
}

async function newChat() {
  try {
    await fetch(`${API}/chat/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `session_id=${encodeURIComponent(sessionId)}`,
    });
  } catch { /* ignore */ }

  sessionId = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, sessionId);
  messagesEl.innerHTML = "";
  setFile(null);
  input.value = "";
  autoGrow();
  updateEmpty();
  updateSendBtn();
  input.focus();
  closeSidebar();
}

function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("overlay").classList.add("hidden");
}

// Events
$("sendBtn").addEventListener("click", send);
$("attachBtn").addEventListener("click", () => fileInput.click());
$("fileRemove").addEventListener("click", () => setFile(null));
$("newChatBtn").addEventListener("click", newChat);

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

input.addEventListener("input", () => {
  autoGrow();
  updateSendBtn();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    input.value = btn.dataset.text || "";
    autoGrow();
    updateSendBtn();
    input.focus();
  });
});

$("menuBtn")?.addEventListener("click", () => {
  $("sidebar").classList.toggle("open");
  $("overlay").classList.toggle("hidden", !$("sidebar").classList.contains("open"));
});

$("overlay")?.addEventListener("click", closeSidebar);

// Drag file onto input box
const inputBox = $("inputBox");
inputBox.addEventListener("dragover", (e) => { e.preventDefault(); });
inputBox.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// Init
checkHealth();
loadHistory();
updateEmpty();
setInterval(checkHealth, 30000);
input.focus();

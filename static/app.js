/**
 * Muhadara RAG — frontend logic.
 * Plain JS + Alpine.js (no build step). Two tabs, two RAG flows.
 */
function muhadaraApp() {
  return {
    tab: 'demo',
    dragHover: false,

    demo: {
      messages: [],     // [{role: 'user'|'assistant', content: str, html?: str}]
      sources:  [],
      input:    '',
      loading:  false,
      examples: [
        'What is NLP and why is it difficult?',
        'ما الفرق بين ال syntax و ال semantics؟',
        'What is ambiguity in natural language?',
        'اشرح ال parsing tree',
        'What is wordnet?',
      ],
    },

    upload: {
      file:        null,
      processing:  false,
      status:      '',
      error:       false,
      transcript:  '',
      summary:     '',
      sessionId:   null,
      numChunks:   0,
      messages:    [],
      input:       '',
      chatLoading: false,
    },

    // ─── Demo lecture tab ─────────────────────────────────
    async askDemo() {
      const q = this.demo.input.trim();
      if (!q || this.demo.loading) return;
      this.demo.input = '';
      this.demo.loading = true;
      this.demo.messages.push({ role: 'user', content: q });
      this.$nextTick(() => this._scroll('demoScroll'));

      try {
        const res  = await fetch('/api/ask', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ question: q }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
        const data = await res.json();
        this.demo.messages.push({
          role:    'assistant',
          content: data.answer,
          html:    this._linkifyTimestamps(data.answer, 'demoAudio'),
        });
        this.demo.sources = data.sources || [];
      } catch (e) {
        this.demo.messages.push({ role: 'assistant', content: `⚠️ Error: ${e.message}` });
      } finally {
        this.demo.loading = false;
        this.$nextTick(() => this._scroll('demoScroll'));
      }
    },

    seekDemoTo(seconds) {
      const audio = this.$refs.demoAudio;
      if (!audio) return;
      audio.currentTime = seconds;
      audio.play().catch(() => {});
    },

    // ─── Upload tab ───────────────────────────────────────
    handleDrop(e) {
      this.dragHover = false;
      const file = e.dataTransfer?.files?.[0];
      if (file && file.type.startsWith('audio')) this.upload.file = file;
    },

    handleFile(e) {
      const file = e.target?.files?.[0];
      if (file) this.upload.file = file;
    },

    async transcribeUpload() {
      if (!this.upload.file || this.upload.processing) return;
      this.upload.processing = true;
      this.upload.status     = 'Uploading + transcribing on GPU …';
      this.upload.error      = false;
      this.upload.transcript = '';
      this.upload.summary    = '';
      this.upload.sessionId  = null;
      this.upload.numChunks  = 0;
      this.upload.messages   = [];

      try {
        const fd = new FormData();
        fd.append('audio', this.upload.file);
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
        const data = await res.json();
        this.upload.transcript = data.transcript;
        this.upload.summary    = data.summary;
        this.upload.sessionId  = data.session_id;
        this.upload.numChunks  = data.num_chunks;
        this.upload.status     = `✅ Transcribed ${data.duration.toFixed(1)}s on ${data.device}.`;
      } catch (e) {
        this.upload.error  = true;
        this.upload.status = `⚠️ ${e.message}`;
      } finally {
        this.upload.processing = false;
      }
    },

    async askUpload() {
      const q = this.upload.input.trim();
      if (!q || this.upload.chatLoading || !this.upload.sessionId) return;
      this.upload.input       = '';
      this.upload.chatLoading = true;
      this.upload.messages.push({ role: 'user', content: q });
      this.$nextTick(() => this._scroll('uploadScroll'));

      try {
        const res = await fetch('/api/ask-upload', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ session_id: this.upload.sessionId, question: q }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
        const data = await res.json();
        this.upload.messages.push({ role: 'assistant', content: data.answer });
      } catch (e) {
        this.upload.messages.push({ role: 'assistant', content: `⚠️ Error: ${e.message}` });
      } finally {
        this.upload.chatLoading = false;
        this.$nextTick(() => this._scroll('uploadScroll'));
      }
    },

    // ─── Helpers ──────────────────────────────────────────
    _scroll(ref) {
      const el = this.$refs[ref];
      if (el) el.scrollTop = el.scrollHeight;
    },

    /** Wrap [MM:SS] or [HH:MM:SS] in clickable buttons that seek the demo audio. */
    _linkifyTimestamps(text, audioRef) {
      const esc = text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return esc.replace(/\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g, (m, ts) => {
        const parts = ts.split(':').map(Number);
        const sec = parts.length === 3
          ? parts[0] * 3600 + parts[1] * 60 + parts[2]
          : parts[0] * 60 + parts[1];
        return `<button onclick="document.querySelector('audio').currentTime=${sec};document.querySelector('audio').play()" class="font-mono text-[11px] px-1.5 py-0.5 bg-white/20 hover:bg-white/30 rounded transition-colors">▶ ${ts}</button>`;
      });
    },
  };
}

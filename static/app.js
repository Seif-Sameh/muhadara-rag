/**
 * Muhadara RAG — unified single-surface chat (Claude/ChatGPT style).
 *
 * Routes /api/ask vs /api/ask-upload based on currentSource.type.
 * File attach button transcribes the clip and switches source in place.
 */
function muhadaraApp() {
  return {
    // ── State ────────────────────────────────────────────
    messages:     [],
    input:        '',
    loading:      false,
    uploading:    false,
    uploadStatus: '',
    uploadError:  false,
    audioPlaying: false,

    currentSource: {
      type:     'demo',                                  // 'demo' | 'upload'
      name:     'NLP Lecture 1 — Syntax & Semantics',
      audioUrl: '/static/demo.mp3',
      sessionId: null,
    },

    examples: [
      'What is NLP and why is it difficult?',
      'ما الفرق بين الـ syntax و الـ semantics؟',
      'Explain ambiguity in natural language',
      'اشرح الـ parsing tree',
      'What is wordnet?',
    ],

    get chatStarted() { return this.messages.length > 0 || this.loading; },

    // ── Composer ─────────────────────────────────────────
    autoGrow(el) {
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    },

    async send() {
      const q = this.input.trim();
      if (!q || this.loading) return;
      this.input = '';
      this.$nextTick(() => {
        this.autoGrow(this.$refs.composer);
        this.autoGrow(this.$refs.composer2);
      });

      this.messages.push({ role: 'user', content: q });
      this.loading = true;
      this.$nextTick(() => this._scrollDown());

      try {
        const data = await this._postAsk(q);
        this.messages.push({
          role:    'assistant',
          content: data.answer,
          html:    this._linkifyTimestamps(data.answer),
          sources: data.sources || [],
        });
      } catch (e) {
        this.messages.push({ role: 'assistant', content: `⚠️ ${e.message}` });
      } finally {
        this.loading = false;
        this.$nextTick(() => this._scrollDown());
      }
    },

    async _postAsk(question) {
      const isUpload = this.currentSource.type === 'upload';
      const url      = isUpload ? '/api/ask-upload' : '/api/ask';
      const body     = isUpload
        ? { session_id: this.currentSource.sessionId, question }
        : { question };

      const res = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json();
    },

    // ── File upload (paperclip button) ───────────────────
    async handleFile(e) {
      const file = e?.target?.files?.[0];
      if (!file) return;
      e.target.value = '';     // allow re-uploading the same file
      await this._upload(file);
    },

    async _upload(file) {
      this.uploading    = true;
      this.uploadError  = false;
      this.uploadStatus = `⏳ Transcribing "${file.name}" on the GPU…`;

      try {
        const fd = new FormData();
        fd.append('audio', file);
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();

        // Swap the active source. Audio playback comes from the local blob.
        const blobUrl = URL.createObjectURL(file);
        if (this._lastBlobUrl) URL.revokeObjectURL(this._lastBlobUrl);
        this._lastBlobUrl = blobUrl;

        this.currentSource = {
          type:      'upload',
          name:      file.name,
          audioUrl:  blobUrl,
          sessionId: data.session_id,
        };
        this.messages = [
          {
            role: 'assistant',
            content:
              `Indexed **${data.num_chunks}** chunk${data.num_chunks === 1 ? '' : 's'} from ` +
              `**${file.name}** (${data.duration.toFixed(1)}s, transcribed on ${data.device}).\n\n` +
              `**Summary:** ${data.summary}\n\n` +
              `Ask me anything about this recording.`,
            html: null,
          },
        ];
        // Render the summary markdown-lite
        this.messages[0].html = this._linkifyTimestamps(this._mdLite(this.messages[0].content));
        this.uploadStatus = `✅ Ready · ${data.num_chunks} chunks · ${data.duration.toFixed(1)}s`;
        this.$nextTick(() => this._scrollDown());
      } catch (e) {
        this.uploadError  = true;
        this.uploadStatus = `⚠️ ${e.message}`;
      } finally {
        this.uploading = false;
        setTimeout(() => { if (!this.uploadError) this.uploadStatus = ''; }, 4000);
      }
    },

    // ── Reset / source switching ─────────────────────────
    resetToDemo() {
      if (this._lastBlobUrl) { URL.revokeObjectURL(this._lastBlobUrl); this._lastBlobUrl = null; }
      this.currentSource = {
        type: 'demo',
        name: 'NLP Lecture 1 — Syntax & Semantics',
        audioUrl: '/static/demo.mp3',
        sessionId: null,
      };
      this.messages = [];
      this.uploadStatus = '';
    },

    resetConversation() {
      this.messages = [];
      this.input    = '';
      this.uploadStatus = '';
    },

    // ── Audio ────────────────────────────────────────────
    toggleAudio() {
      const a = this.$refs.audio;
      if (!a) return;
      if (a.paused) a.play().catch(() => {});
      else a.pause();
    },

    seekTo(seconds) {
      const a = this.$refs.audio;
      if (!a) return;
      a.currentTime = seconds;
      a.play().catch(() => {});
    },

    // ── Helpers ──────────────────────────────────────────
    _scrollDown() {
      const el = this.$refs.scroll;
      if (el) el.scrollTop = el.scrollHeight;
    },

    /** Tiny markdown for bold (**text**) and newlines. Just for the system message. */
    _mdLite(text) {
      return text
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    },

    /** Wrap [MM:SS] / [HH:MM:SS] in clickable chips that seek the current audio. */
    _linkifyTimestamps(text) {
      // If we already escaped via _mdLite, this just operates on the resulting string.
      // Otherwise escape first.
      let s = text;
      if (!/<strong>/.test(s)) {
        s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      }
      return s.replace(/\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g, (_, ts) => {
        const parts = ts.split(':').map(Number);
        const sec   = parts.length === 3
          ? parts[0] * 3600 + parts[1] * 60 + parts[2]
          : parts[0] * 60 + parts[1];
        return `<button onclick="(function(){const a=document.querySelector('audio');if(a){a.currentTime=${sec};a.play().catch(()=>{});}})()" class="inline-flex items-center gap-1 font-mono text-[11px] px-1.5 py-0.5 mx-0.5 bg-emerald-500/15 text-emerald-200 hover:bg-emerald-500/25 rounded ring-1 ring-emerald-400/20 transition-colors align-middle" dir="ltr">▶ ${ts}</button>`;
      });
    },
  };
}

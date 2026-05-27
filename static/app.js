/**
 * Muhadara RAG — unified single-surface chat.
 *
 *  - sessionStorage persistence (refresh keeps your chat)
 *  - Markdown rendering for LLM answers (marked + DOMPurify)
 *  - [MM:SS] citations become clickable chips (data-seek + event delegation)
 *  - File attach transcribes a clip and switches the active source in place
 *  - Stop button (AbortController) interrupts an in-flight request
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
    _abort:       null,

    currentSource: {
      type:      'demo',
      name:      'NLP Lecture 1 — Syntax & Semantics',
      audioUrl:  '/static/demo.mp3',
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

    // ── Persistence (sessionStorage) ─────────────────────
    _key: 'muhadara.session.v4',

    restore() {
      try {
        const raw = sessionStorage.getItem(this._key);
        if (!raw) return;
        const s = JSON.parse(raw);
        // Never restore a stale upload session (server may have lost it on restart).
        if (s.currentSource?.type === 'upload') return;
        if (Array.isArray(s.messages)) this.messages = s.messages;
        if (s.currentSource)            this.currentSource = s.currentSource;
        this.$nextTick(() => this._scrollDown());
      } catch {}
      this.$watch('messages', () => this._persist());
      this.$watch('currentSource', () => this._persist());
    },

    _persist() {
      try {
        sessionStorage.setItem(this._key, JSON.stringify({
          messages:      this.messages,
          currentSource: this.currentSource,
        }));
      } catch {}
    },

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

      this.messages.push({ role: 'user', content: q, html: this._escape(q) });
      this.loading = true;
      this.$nextTick(() => this._scrollDown());

      this._abort = new AbortController();
      try {
        const data = await this._postAsk(q, this._abort.signal);
        this.messages.push({
          role:        'assistant',
          content:     data.answer,
          html:        this._renderAnswer(data.answer),
          sources:     data.sources || [],
          sourcesOpen: false,
          copied:      false,
        });
      } catch (e) {
        if (e.name === 'AbortError') {
          this.messages.push({ role: 'assistant', content: '⏹  Stopped.', html: '<em class="text-zinc-500">Stopped.</em>' });
        } else {
          this.messages.push({ role: 'assistant', content: `⚠️ ${e.message}`, html: this._escape(`⚠️ ${e.message}`) });
        }
      } finally {
        this.loading = false;
        this._abort  = null;
        this.$nextTick(() => this._scrollDown());
      }
    },

    stopGeneration() {
      if (this._abort) this._abort.abort();
    },

    async _postAsk(question, signal) {
      const isUpload = this.currentSource.type === 'upload';
      const url      = isUpload ? '/api/ask-upload' : '/api/ask';
      const body     = isUpload
        ? { session_id: this.currentSource.sessionId, question }
        : { question };

      const res = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json();
    },

    // ── Upload ───────────────────────────────────────────
    async handleFile(e) {
      const file = e?.target?.files?.[0];
      if (!file) return;
      e.target.value = '';
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

        const blobUrl = URL.createObjectURL(file);
        if (this._lastBlobUrl) URL.revokeObjectURL(this._lastBlobUrl);
        this._lastBlobUrl = blobUrl;

        this.currentSource = {
          type:      'upload',
          name:      file.name,
          audioUrl:  blobUrl,
          sessionId: data.session_id,
        };
        const intro =
          `**Indexed ${data.num_chunks} chunk${data.num_chunks === 1 ? '' : 's'} from \`${file.name}\`** ` +
          `(${data.duration.toFixed(1)}s, transcribed on ${data.device}).\n\n` +
          `**Summary:** ${data.summary}\n\n` +
          `Ask me anything about this recording.`;
        this.messages = [{
          role:        'assistant',
          content:     intro,
          html:        this._renderAnswer(intro),
          sources:     [],
          sourcesOpen: false,
          copied:      false,
        }];
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

    // ── Source switching ─────────────────────────────────
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

    newChat() {
      this.messages = [];
      this.input    = '';
      this.uploadStatus = '';
      this.$nextTick(() => this.$refs.composer?.focus());
    },

    // ── Audio + click delegation ─────────────────────────
    toggleAudio() {
      const a = this.$refs.audio;
      if (!a) return;
      if (a.paused) a.play().catch(() => {});
      else          a.pause();
    },

    seekTo(seconds) {
      const a = this.$refs.audio;
      if (!a) return;
      a.currentTime = +seconds;
      a.play().catch(() => {});
    },

    /** Delegated click handler — any element with [data-seek] seeks the audio.
        Used by inline timestamp chips (rendered into LLM answers + source cards). */
    onMessagesClick(e) {
      const el = e.target.closest('[data-seek]');
      if (el) this.seekTo(el.dataset.seek);
    },

    // ── Copy ─────────────────────────────────────────────
    async copyMessage(msg, _e) {
      try {
        await navigator.clipboard.writeText(msg.content);
        msg.copied = true;
        setTimeout(() => { msg.copied = false; }, 1500);
      } catch {}
    },

    // ── Helpers ──────────────────────────────────────────
    _scrollDown() {
      const el = this.$refs.scroll;
      if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    },

    _escape(t) {
      return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    },

    /** Full assistant answer rendering: markdown → timestamp chips → sanitize. */
    _renderAnswer(text) {
      // 1. Replace [MM:SS] with clickable chip markup BEFORE markdown so marked
      //    treats the chip <button> as inline HTML.
      const chipped = String(text).replace(
        /\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g,
        (_, ts) => {
          const parts = ts.split(':').map(Number);
          const sec   = parts.length === 3
            ? parts[0] * 3600 + parts[1] * 60 + parts[2]
            : parts[0] * 60 + parts[1];
          return `<button data-seek="${sec}" class="inline-flex items-center gap-1 font-mono text-[11px] px-1.5 py-0.5 mx-0.5 bg-emerald-500/15 text-emerald-200 hover:bg-emerald-500/25 rounded ring-1 ring-emerald-400/20 transition-colors align-middle" dir="ltr">▶ ${ts}</button>`;
        }
      );

      const rawHtml = (typeof marked !== 'undefined')
        ? marked.parse(chipped, { breaks: true, gfm: true })
        : this._escape(chipped);

      if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(rawHtml, {
          ADD_ATTR: ['data-seek', 'dir'],
        });
      }
      return rawHtml;
    },
  };
}

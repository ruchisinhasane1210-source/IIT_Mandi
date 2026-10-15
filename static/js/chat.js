/* TraceLearn — tutor chat client */
(function () {
  const scroll = document.getElementById('chat-scroll');
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const send = document.getElementById('chat-send');
  const preview = document.getElementById('source-preview');
  const courseId = (window.TL && window.TL.courseId) || 'dsa_6006';

  const BADGE = {
    grounded: 'badge-high',
    partial: 'badge-mid',
    not_covered: 'badge-low',
  };

  function el(html) {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  function esc(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function citeLabel(c) {
    if (c.source_type === 'video' && c.timestamp_start != null) return 'Play from ' + hms(c.timestamp_start);
    if (c.source_type === 'slide' && c.slide_number != null) return 'Open slide ' + c.slide_number;
    if (c.page_number != null) return 'Open page ' + c.page_number;
    if (c.figure_reference) return 'Open figure';
    return 'Open source';
  }

  function hms(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
  }

  function iconSvg() {
    return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 12.5 9 17.5 20 6.5"/></svg>';
  }

  function addUser(text) {
    scroll.appendChild(el(`<div class="bubble bubble-user">${esc(text)}</div>`));
    scroll.scrollTop = scroll.scrollHeight;
  }

  function addTyping() {
    const node = el('<div class="bubble bubble-bot" id="typing"><span class="typing"><i></i><i></i><i></i></span></div>');
    scroll.appendChild(node);
    scroll.scrollTop = scroll.scrollHeight;
    return node;
  }

  function addBot(result) {
    const status = result.grounding_status || 'grounded';
    const cites = (result.citations || []).map((c) => (
      `<button class="cite" data-unit="${c.id}">
         <span>${esc(citeLabel(c))}</span>
         <span class="cite-type">${esc(c.source_id || '')}</span>
         <span class="cite-score">${(c.score != null ? c.score : '')}</span>
       </button>`
    )).join('');

    const node = el(`
      <div class="bubble bubble-bot">
        <div>${esc(result.answer).replace(/\n/g, '<br>')}</div>
        <div class="bubble-meta">
          <span class="badge ${BADGE[status] || 'badge'}"><span class="dot"></span>${esc(status.replace('_', ' '))}</span>
          <span class="mono">${result.grounding_score != null ? result.grounding_score : ''}</span>
        </div>
        ${cites ? `<div class="cite-list">${cites}</div>` : ''}
        ${result.outside_knowledge ? `<div class="note warn mt-sm">${esc(result.outside_knowledge)}</div>` : ''}
      </div>`);
    scroll.appendChild(node);
    scroll.scrollTop = scroll.scrollHeight;
  }

  async function openCitation(unitId, button) {
    preview.innerHTML = '<div class="skel" style="height:14px;margin-bottom:10px"></div><div class="skel" style="height:14px;width:80%"></div>';
    try {
      const res = await fetch(`/api/units/${unitId}`);
      const data = await res.json();
      const loc = data.location || {};
      preview.innerHTML = `
        <div class="row between" style="margin-bottom:10px">
          <span class="badge badge-primary">${esc(data.source_type || 'source')}</span>
          <span class="tiny muted">${esc(data.source_id || '')}</span>
        </div>
        <div>${esc(data.content || '')}</div>
        <div class="divider"></div>
        <a class="btn btn-outline btn-sm" href="${data.jump_url || '#'}" target="_blank" rel="noopener">
          ${esc(loc.label || 'Open source')}
        </a>`;
      if (button) {
        document.querySelectorAll('.cite').forEach((b) => b.style.borderColor = '');
        button.style.borderColor = 'var(--primary)';
      }
    } catch (e) {
      preview.innerHTML = '<div class="small muted">Could not load that source.</div>';
    }
  }

  async function ask(message) {
    addUser(message);
    const typing = addTyping();
    send.disabled = true;
    try {
      const res = await fetch(`/api/courses/${courseId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      typing.remove();
      addBot(data);
    } catch (e) {
      typing.remove();
      addBot({ answer: 'Network error — is the server still running?', grounding_status: 'not_covered' });
    } finally {
      send.disabled = false;
      input.focus();
    }
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    const value = input.value.trim();
    if (!value) return;
    input.value = '';
    ask(value);
  });

  document.addEventListener('click', function (event) {
    const cite = event.target.closest('.cite');
    if (cite) { openCitation(cite.dataset.unit, cite); return; }
    const chip = event.target.closest('.chip[data-prompt]');
    if (chip) { input.value = chip.dataset.prompt; input.focus(); }
  });

  scroll.scrollTop = scroll.scrollHeight;
})();

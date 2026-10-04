/* Zero Sum: offline-first UI for the exported, trained Q-learning policy.
   The policy is fixed during play. There is no search/minimax in this client. */
(() => {
  'use strict';
  const WIN_LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  const LABELS = ['top left','top middle','top right','middle left','center','middle right','bottom left','bottom middle','bottom right'];
  const SCORE_KEY = 'zero-sum-qlearning-scores-v1';
  const $ = id => document.getElementById(id);
  const cells = [...document.querySelectorAll('.cell')];
  const sideButtons = [...document.querySelectorAll('.side-option')];
  const marks = {
    X: '<svg viewBox="0 0 64 64" aria-hidden="true"><path d="M13 13 51 51M51 13 13 51"/></svg>',
    O: '<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="22"/></svg>'
  };
  let model = null;
  let board = Array(9).fill('');
  let human = 'X';
  let opponent = 'trained';
  let turn = 'X';
  let ready = false;
  let busy = false;
  let ended = false;
  let winLine = [];
  let outcome = null;
  let round = 1;
  let generation = 0;
  let agentTimer = null;
  let toastTimer = null;
  let lastAgentMove = -1;
  let lastDecision = null;
  let scores = loadScores();

  function loadScores() {
    try {
      const parsed = JSON.parse(localStorage.getItem(SCORE_KEY) || '{}');
      const result = {};
      for (const key of ['trained', 'random', 'fallback']) {
        const value = parsed?.[key];
        result[key] = { wins: validCount(value?.wins), draws: validCount(value?.draws), losses: validCount(value?.losses) };
      }
      return result;
    } catch { return { trained: emptyScore(), random: emptyScore(), fallback: emptyScore() }; }
  }
  function validCount(value) { return Number.isSafeInteger(value) && value >= 0 ? value : 0; }
  function emptyScore() { return { wins: 0, draws: 0, losses: 0 }; }
  function saveScores() { try { localStorage.setItem(SCORE_KEY, JSON.stringify(scores)); } catch { /* Private/sandboxed browsers can block storage; games still work. */ } }
  function scoreBucket() { return opponent === 'random' ? 'random' : model ? 'trained' : 'fallback'; }
  function renderScores() {
    const score = scores[scoreBucket()];
    for (const key of ['wins', 'draws', 'losses']) $(key).textContent = score[key];
    $('score-mode').textContent = opponent === 'random' ? 'Against the random baseline' : model ? 'Against the trained agent' : 'Against the random fallback';
  }
  function agentMark() { return human === 'X' ? 'O' : 'X'; }
  function legalMoves(position) { return position.map((mark, i) => mark ? -1 : i).filter(i => i !== -1); }
  function encodeState(position) { return position.map(mark => !mark ? '0' : mark === agentMark() ? '1' : '2').join(''); }
  function terminal(position) {
    for (const line of WIN_LINES) if (position[line[0]] && line.every(i => position[i] === position[line[0]])) return { winner: position[line[0]], line };
    return position.every(Boolean) ? { winner: null, line: [] } : null;
  }
  function randomItem(items) { return items[Math.floor(Math.random() * items.length)]; }
  function qRow(state) {
    const row = model?.q_table?.[state];
    return Array.isArray(row) && row.length === 9 ? row : null;
  }
  function chooseAgentAction() {
    const legal = legalMoves(board);
    const state = encodeState(board);
    const values = qRow(state);
    const learned = opponent === 'trained' && !!values && legal.every(i => Number.isFinite(values[i]));
    if (!learned) {
      const action = randomItem(legal);
      return { action, state, values: null, legal, best: [], source: opponent === 'random' ? 'random' : 'fallback' };
    }
    const maximum = Math.max(...legal.map(i => values[i]));
    const best = legal.filter(i => Math.abs(values[i] - maximum) <= 1e-12);
    return { action: randomItem(best), state, values, legal, best, source: 'learned' };
  }

  function renderBoard() {
    const playable = ready && !busy && !ended && turn === human;
    const legal = legalMoves(board);
    const focusIndex = legal[0] ?? 0;
    cells.forEach((cell, i) => {
      const mark = board[i];
      const owner = mark ? mark === human ? 'human' : 'agent' : '';
      cell.className = `cell${owner ? ' ' + owner : ''}${winLine.includes(i) ? ' win-' + owner : ''}${i === lastAgentMove ? ' last-move' : ''}`;
      // Preserve existing SVG nodes so a new move does not replay every mark animation.
      if (cell.dataset.mark !== mark) { cell.innerHTML = mark ? marks[mark] : ''; cell.dataset.mark = mark; }
      cell.disabled = !playable || !!mark;
      cell.tabIndex = playable && i === focusIndex ? 0 : -1;
      cell.setAttribute('aria-label', `Row ${Math.floor(i / 3) + 1}, column ${i % 3 + 1}, ${mark ? mark + ', ' + (owner === 'human' ? 'you' : 'agent') : 'empty'}`);
    });
    $('human-mark-label').textContent = human;
    $('agent-mark-label').textContent = agentMark();
    $('round-label').textContent = `ROUND ${String(round).padStart(2, '0')}`;
    sideButtons.forEach(button => {
      const selected = button.dataset.mark === human;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    renderStatus();
  }
  function renderStatus() {
    let title, detail, symbol, theme = '';
    if (!ready) {
      title = 'Getting the board ready'; detail = 'Loading the learned Q-table…'; symbol = '·';
    } else if (ended) {
      if (!outcome.winner) { title = 'Perfectly balanced.'; detail = 'A draw. Nine squares, no winner. Go again?'; symbol = '='; theme = 'draw'; }
      else if (outcome.winner === human) { title = 'That round is yours.'; detail = 'Nicely played. Ready for a rematch?'; symbol = '✓'; }
      else { title = 'The agent takes this one.'; detail = 'Every opponent has a pattern. Try another round.'; symbol = agentMark() === 'X' ? '✕' : '○'; theme = 'agent'; }
    } else if (busy || turn !== human) {
      title = 'The agent is choosing…'; detail = opponent === 'random' ? 'Sampling a legal move at random.' : model ? 'Reading the board. Comparing learned values.' : 'Using the random fallback; policy unavailable.'; symbol = agentMark() === 'X' ? '✕' : '○'; theme = 'agent';
    } else {
      title = 'Your move.'; detail = `You’re playing ${human}. Pick an empty square.`; symbol = human === 'X' ? '✕' : '○';
    }
    $('game-status').textContent = title;
    $('game-status-detail').textContent = detail;
    $('turn-indicator').textContent = symbol;
    $('turn-indicator').className = 'turn-indicator' + (theme ? ' ' + theme : '');
  }
  function finishIfTerminal() {
    const result = terminal(board);
    if (!result) return false;
    if (!ended) {
      ended = true; busy = false; outcome = result; winLine = result.line;
      scores[scoreBucket()][!result.winner ? 'draws' : result.winner === human ? 'wins' : 'losses'] += 1;
      saveScores(); renderScores();
    }
    renderBoard();
    return true;
  }
  function humanMove(index) {
    if (!ready || busy || ended || turn !== human || board[index]) return;
    board[index] = human;
    if (finishIfTerminal()) return;
    turn = agentMark();
    scheduleAgent();
  }
  function scheduleAgent() {
    if (!ready || ended || turn === human) return;
    busy = true; renderBoard();
    const ticket = generation;
    clearTimeout(agentTimer);
    agentTimer = setTimeout(() => {
      if (ticket !== generation || ended || turn === human) return;
      const decision = chooseAgentAction();
      if (!Number.isInteger(decision.action) || board[decision.action]) return;
      lastDecision = decision; lastAgentMove = decision.action;
      board[decision.action] = agentMark(); busy = false;
      renderInsight();
      if (finishIfTerminal()) return;
      turn = human; renderBoard();
    }, window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 160 : 460);
  }
  function newRound(increment = true) {
    generation += 1;
    clearTimeout(agentTimer);
    if (increment) round += 1;
    board = Array(9).fill(''); turn = 'X'; busy = false; ended = false;
    winLine = []; outcome = null; lastAgentMove = -1; lastDecision = null;
    renderBoard(); renderInsight(); renderScores();
    if (ready && turn !== human) scheduleAgent();
  }
  function renderInsight() {
    const visible = $('show-q').checked;
    $('q-insight').hidden = !visible;
    if (!visible) return;
    $('q-values').replaceChildren();
    if (!lastDecision) {
      $('q-title').textContent = 'Waiting for the agent';
      $('q-caption').textContent = 'The next agent move will reveal a snapshot of its legal action values.';
      return;
    }
    const { action, values, legal, best, source } = lastDecision;
    $('q-title').textContent = source === 'learned' ? 'Last decision · learned policy' : source === 'random' ? 'Last decision · random baseline' : 'Last decision · random fallback';
    for (let i = 0; i < 9; i++) {
      const value = document.createElement('div');
      value.className = 'q-value' + (best.includes(i) ? ' best' : '') + (!legal.includes(i) ? ' unavailable' : '');
      value.textContent = legal.includes(i) && values ? (values[i] >= 0 ? '+' : '') + values[i].toFixed(3) : '—';
      const label = document.createElement('small');
      label.textContent = `${i + 1}${i === action ? ' · played' : !legal.includes(i) ? ' · occupied' : ''}`;
      value.appendChild(label);
      value.setAttribute('aria-label', `${LABELS[i]}: ${!legal.includes(i) ? 'occupied' : values ? 'Q value ' + values[i].toFixed(3) : 'no Q value'}${i === action ? ', selected' : ''}`);
      $('q-values').appendChild(value);
    }
    $('q-caption').textContent = source === 'learned'
      ? `Chose ${LABELS[action]}. ${best.length > 1 ? `${best.length} moves shared the highest value; ties are broken randomly.` : 'It had the highest learned value among legal moves.'} Values are from before that move.`
      : source === 'random'
        ? `Chose ${LABELS[action]} uniformly from ${legal.length} legal moves. The random baseline does not consult the Q-table.`
        : `Chose ${LABELS[action]} at random. ${model ? 'This position is missing from the learned table.' : 'The learned policy is unavailable.'} No search or hidden solver is used.`;
  }
  function announce(message) {
    clearTimeout(toastTimer); $('toast').textContent = message; $('toast').classList.add('visible');
    toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 2600);
  }
  function shortNumber(value) { return new Intl.NumberFormat('en', { notation: value >= 10000 ? 'compact' : 'standard', maximumFractionDigits: 1 }).format(value); }
  function metadataNumber(metadata, ...names) {
    for (const name of names) { const value = metadata?.[name]; if (typeof value === 'number' && Number.isFinite(value)) return value; }
    return null;
  }
  function showModelDetails() {
    $('model-badge').classList.toggle('warning', !model);
    $('model-badge-text').textContent = model ? 'Learned policy · ready' : 'Random fallback';
    if (model) {
      const metadata = model.metadata || {};
      const episodes = metadataNumber(metadata, 'episodes', 'training_episodes', 'num_episodes', 'number_of_episodes');
      $('episodes-value').textContent = episodes === null ? 'See report' : shortNumber(episodes);
      $('states-value').textContent = shortNumber(Object.keys(model.q_table).length);
      $('episodes-value').title = episodes === null ? 'Episode count is not included in this policy file.' : episodes.toLocaleString('en') + ' training episodes';
      $('states-value').title = Object.keys(model.q_table).length.toLocaleString('en') + ' exported board positions, including rotations and reflections';
      $('settings-note').textContent = 'A learned policy. A fresh challenge. Runs offline.';
      const rewards = metadata.rewards;
      if (rewards && typeof rewards === 'object') {
        const entries = ['win', 'draw', 'loss', 'step'].filter(key => typeof rewards[key] === 'number').map(key => `${key}: ${rewards[key] > 0 ? '+' : ''}${rewards[key]}`);
        if (entries.length) $('reward-explanation').textContent = 'Training rewards — ' + entries.join(' · ') + '. See the notebook and report for the complete training setup.';
      }
    } else {
      $('episodes-value').textContent = '—'; $('states-value').textContent = '—';
      $('settings-note').textContent = 'Policy unavailable. The game uses a clearly labeled random fallback.';
      $('insight-description').textContent = 'The trained model could not be loaded. Place artifacts/model.js beside the project files, or open TicTacToe_Offline.html.';
    }
  }

  cells.forEach((cell, index) => {
    cell.addEventListener('click', () => humanMove(index));
    cell.addEventListener('keydown', event => {
      const offsets = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 3, ArrowUp: -3 };
      if (!(event.key in offsets) && event.key !== 'Home' && event.key !== 'End') return;
      event.preventDefault();
      const legal = legalMoves(board).filter(i => !cells[i].disabled);
      if (!legal.length) return;
      let destination = index;
      if (event.key === 'Home') destination = legal[0];
      else if (event.key === 'End') destination = legal[legal.length - 1];
      else {
        const step = offsets[event.key];
        for (let tries = 0; tries < 9; tries++) { destination = (destination + step + 9) % 9; if (legal.includes(destination)) break; }
        // A full occupied column can block vertical navigation; choose another available square.
        if (destination === index) destination = legal[(legal.indexOf(index) + 1) % legal.length];
      }
      cells.forEach((item, i) => { item.tabIndex = i === destination ? 0 : -1; });
      cells[destination].focus();
    });
  });
  sideButtons.forEach(button => button.addEventListener('click', () => {
    if (human === button.dataset.mark) return;
    human = button.dataset.mark; newRound(board.some(Boolean));
    announce(`You’re playing ${human}. ${human === 'X' ? 'You go first.' : 'The agent goes first.'}`);
  }));
  $('new-game').addEventListener('click', () => { newRound(); announce('A fresh board. Your next challenge.'); });
  $('show-q').addEventListener('change', renderInsight);
  $('opponent').addEventListener('change', event => {
    opponent = event.target.value;
    $('settings-note').textContent = opponent === 'random' ? 'A uniform random opponent. Compare it with the learned policy.' : model ? 'A learned policy. A fresh challenge. Runs offline.' : 'Policy unavailable. The game uses a clearly labeled random fallback.';
    newRound(board.some(Boolean));
    announce(opponent === 'random' ? 'Random baseline selected.' : model ? 'Trained Q-agent selected.' : 'Policy unavailable. Random fallback active.');
  });
  $('reset-score').addEventListener('click', () => { scores[scoreBucket()] = emptyScore(); saveScores(); renderScores(); announce('Scorecard reset for this opponent.'); });

  async function initialize() {
    renderBoard(); renderScores();
    try {
      let loaded = window.TICTACTOE_MODEL;
      if (!loaded && location.protocol !== 'file:') {
        const response = await fetch('artifacts/q_table.json');
        if (!response.ok) throw new Error('Could not load policy');
        loaded = await response.json();
      }
      if (!loaded?.q_table || typeof loaded.q_table !== 'object' || !Object.keys(loaded.q_table).length) throw new Error('Policy has no states');
      model = loaded;
    } catch (error) { console.warn('Q-learning policy unavailable; using explicitly labeled random fallback.', error); }
    ready = true; showModelDetails(); newRound(false);
  }
  initialize();
})();

/* Dependency-free interaction tests against the real app.js using a small DOM
   harness. Run: node --test tests/frontend.test.cjs */
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'app.js'), 'utf8');

function harness(table = {}, storageBlocked = false) {
  let focused = null;
  class Element {
    constructor() { this.dataset = {}; this.handlers = {}; this.className = ''; this.children = []; this.textContent = ''; this.checked = false; this.attributes = {}; }
    get classList() {
      return {
        add: value => { this.className += ' ' + value; },
        remove: value => { this.className = this.className.split(' ').filter(item => item !== value).join(' '); },
        toggle: (value, force) => { const set = new Set(this.className.split(' ').filter(Boolean)); if (force) set.add(value); else set.delete(value); this.className = [...set].join(' '); }
      };
    }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(type, handler) { this.handlers[type] = handler; }
    fire(type, extra = {}) { this.handlers[type]?.({ target: this, preventDefault() {}, ...extra }); }
    appendChild(child) { this.children.push(child); }
    replaceChildren() { this.children = []; }
    focus() { focused = this; }
  }
  const nodes = new Map();
  const get = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  const cells = Array.from({ length: 9 }, (_, i) => { const element = new Element(); element.dataset.index = String(i); return element; });
  const sides = ['X', 'O'].map(mark => { const element = new Element(); element.dataset.mark = mark; return element; });
  let id = 0;
  const timers = new Map();
  const document = { getElementById: get, querySelectorAll: selector => selector === '.cell' ? cells : sides, createElement: () => new Element() };
  const q_table = Object.keys(table).length ? table : { '000000000': Array(9).fill(0) };
  const window = { TICTACTOE_MODEL: { metadata: { episodes: 160000 }, q_table }, matchMedia: () => ({ matches: false }) };
  const context = { window, document, location: { protocol: 'file:' }, console, Intl, Math: Object.create(Math),
    localStorage: { getItem() { if (storageBlocked) throw Error('blocked'); return null; }, setItem() { if (storageBlocked) throw Error('blocked'); } },
    setTimeout: (callback, delay) => { timers.set(++id, { callback, delay }); return id; }, clearTimeout: timerId => timers.delete(timerId) };
  context.Math.random = () => 0;
  vm.runInNewContext(source, context, { filename: 'app.js' });
  return { cells, sides, get, timers, focused: () => focused,
    flushAgent() { for (const [timerId, timer] of timers) if (timer.delay <= 460) { timers.delete(timerId); timer.callback(); break; } },
    marks() { return cells.map(cell => cell.dataset.mark); } };
}
function row(action, value = 1) { const result = Array(9).fill(0); result[action] = value; return result; }

test('greedy selection masks occupied squares and rejects clicks while the agent is pending', () => {
  const values = row(4); values[0] = 999;
  const game = harness({ '200000000': values });
  game.cells[0].fire('click'); game.cells[1].fire('click');
  assert.equal(game.marks().filter(Boolean).length, 1);
  game.flushAgent();
  assert.equal(game.marks()[4], 'O');
  assert.equal(game.marks()[1], '');
  assert.equal(game.get('game-status').textContent, 'Your move.');
});

test('restart invalidates even an already captured old agent callback', () => {
  const game = harness({ '200000000': row(4) });
  game.cells[0].fire('click');
  const stale = [...game.timers.values()].find(timer => timer.delay === 460).callback;
  game.get('new-game').fire('click');
  stale();
  assert.equal(game.marks().filter(Boolean).length, 0);
  assert.equal(game.get('round-label').textContent, 'ROUND 02');
});

test('choosing O lets agent X start and restarts the board', () => {
  const game = harness({ '000000000': row(4) });
  game.sides[1].fire('click');
  assert.ok(game.cells.every(cell => cell.disabled));
  game.flushAgent();
  assert.equal(game.marks()[4], 'X');
  assert.equal(game.get('human-mark-label').textContent, 'O');
  game.cells[0].fire('click');
  assert.equal(game.marks()[0], 'O');
});

test('terminal human win is counted once; finished board ignores further actions', () => {
  const game = harness();
  game.get('opponent').value = 'random'; game.get('opponent').fire('change');
  game.cells[0].fire('click'); game.flushAgent();
  game.cells[3].fire('click'); game.flushAgent();
  game.cells[6].fire('click');
  assert.equal(game.get('wins').textContent, 1);
  assert.equal(game.get('game-status').textContent, 'That round is yours.');
  game.cells[8].fire('click');
  assert.equal(game.get('wins').textContent, 1);
  assert.equal(game.marks()[8], '');
  assert.ok(game.cells[0].className.includes('win-human'));
  game.get('opponent').value = 'trained'; game.get('opponent').fire('change');
  assert.equal(game.get('wins').textContent, 0, 'scores are separate for each opponent');
});

test('terminal agent win increments agent score', () => {
  const game = harness({ '200000000': row(1), '212000000': row(4), '212010200': row(7) });
  for (const action of [0, 2, 6]) { game.cells[action].fire('click'); game.flushAgent(); }
  assert.equal(game.get('losses').textContent, 1);
  assert.equal(game.get('game-status').textContent, 'The agent takes this one.');
});

test('unseen-state fallback is labeled and Q insight never invents values', () => {
  const game = harness();
  game.get('show-q').checked = true; game.get('show-q').fire('change');
  game.cells[0].fire('click'); game.flushAgent();
  assert.equal(game.get('q-title').textContent, 'Last decision · random fallback');
  assert.match(game.get('q-caption').textContent, /missing from the learned table/);
  assert.ok(game.get('q-values').children.every(child => child.textContent === '—'));
});

test('keyboard navigation works and blocked localStorage does not prevent playing', () => {
  const game = harness({}, true);
  game.cells[0].fire('keydown', { key: 'ArrowRight' });
  assert.equal(game.focused(), game.cells[1]);
  game.cells[1].fire('click'); game.flushAgent();
  assert.equal(game.marks().filter(Boolean).length, 2);
});

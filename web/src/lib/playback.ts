/**
 * requestAnimationFrame playback clock over replay frame indices.
 *
 * `t` is a fractional frame index. It advances by real elapsed time times
 * (fps / step) times speed, so playback speed is independent of display
 * refresh rate. Listeners are called synchronously from the rAF callback.
 * Nothing here touches React.
 */

export type TickListener = (t: number) => void;
export type StateListener = (playing: boolean) => void;

export class Playback {
  t = 0;
  playing = false;
  speed = 1;
  private n: number;
  private framesPerSecond: number;
  private raf = 0;
  private last = 0;
  private tickers = new Set<TickListener>();
  private states = new Set<StateListener>();
  private readonly now: () => number;
  private readonly raff: (cb: FrameRequestCallback) => number;
  private readonly caf: (h: number) => void;

  constructor(
    nFrames: number,
    fps: number,
    step = 1,
    deps: { now?: () => number; raf?: (cb: FrameRequestCallback) => number; caf?: (h: number) => void } = {},
  ) {
    this.n = Math.max(1, nFrames);
    this.framesPerSecond = fps / Math.max(1, step);
    this.now = deps.now ?? (() => performance.now());
    this.raff = deps.raf ?? ((cb) => requestAnimationFrame(cb));
    this.caf = deps.caf ?? ((h) => cancelAnimationFrame(h));
  }

  get lastIndex(): number {
    return this.n - 1;
  }

  onTick(fn: TickListener): () => void {
    this.tickers.add(fn);
    return () => this.tickers.delete(fn);
  }

  onState(fn: StateListener): () => void {
    this.states.add(fn);
    return () => this.states.delete(fn);
  }

  play(): void {
    if (this.playing) return;
    if (this.t >= this.n - 1) this.t = 0;
    this.playing = true;
    this.last = this.now();
    this.states.forEach((f) => f(true));
    this.raf = this.raff(this.loop);
  }

  /** Pause and snap to the nearest whole frame, so analysis matches what is drawn. */
  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    if (this.raf) this.caf(this.raf);
    this.raf = 0;
    this.t = Math.round(this.t);
    this.emit();
    this.states.forEach((f) => f(false));
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(t: number): void {
    this.t = Math.max(0, Math.min(this.n - 1, t));
    this.emit();
  }

  step(delta: number): void {
    if (this.playing) this.pause();
    this.seek(Math.round(this.t) + delta);
  }

  setSpeed(s: number): void {
    this.speed = s;
  }

  destroy(): void {
    if (this.raf) this.caf(this.raf);
    this.raf = 0;
    this.playing = false;
    this.tickers.clear();
    this.states.clear();
  }

  private emit(): void {
    this.tickers.forEach((f) => f(this.t));
  }

  private loop = (): void => {
    if (!this.playing) return;
    const now = this.now();
    // clamp so a backgrounded tab does not jump far ahead on return
    const dt = Math.min(0.1, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    this.t += dt * this.framesPerSecond * this.speed;
    if (this.t >= this.n - 1) {
      this.t = this.n - 1;
      this.emit();
      this.pause();
      return;
    }
    this.emit();
    this.raf = this.raff(this.loop);
  };
}

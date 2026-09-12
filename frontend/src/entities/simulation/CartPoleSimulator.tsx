import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';
import { Button } from '@/shared/ui/Button';
import { Slider } from '@/shared/ui/Slider';
import { Badge } from '@/shared/ui/Badge';

export const CartPoleSimulator: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Physics state
  const [running, setRunning] = useState(false);
  const [autoBalance, setAutoBalance] = useState(true);

  // Parameters
  const gravity = 9.8;
  const [massCart, setMassCart] = useState(1.0);
  const [massPole, setMassPole] = useState(0.1);
  const [length, setLength] = useState(0.5); // half-pole length
  const [forceMag, setForceMag] = useState(10.0);

  // Dynamic variables
  const stateRef = useRef({
    x: 0, // cart position
    xDot: 0, // cart velocity
    theta: 0.05, // pole angle (radians)
    thetaDot: 0, // angular velocity
    stepCount: 0,
    force: 0,
  });

  const [metrics, setMetrics] = useState({
    x: 0,
    thetaDeg: 0,
    steps: 0,
    status: 'Ready',
  });

  const resetSimulation = () => {
    stateRef.current = {
      x: 0,
      xDot: 0,
      theta: 0.05,
      thetaDot: 0,
      stepCount: 0,
      force: 0,
    };
    setMetrics({
      x: 0,
      thetaDeg: (0.05 * 180) / Math.PI,
      steps: 0,
      status: 'Reset',
    });
  };

  useEffect(() => {
    let animationFrameId: number;
    let lastTime = performance.now();

    const dt = 0.02; // 50 Hz physics step

    const updatePhysics = () => {
      const s = stateRef.current;

      // Auto PID/LQR balancer policy
      if (autoBalance) {
        // Simple PD controller on theta and x
        const kpTheta = 50.0;
        const kdTheta = 10.0;
        const kpX = 2.0;
        const kdX = 3.0;

        const actionForce =
          kpTheta * s.theta + kdTheta * s.thetaDot + kpX * s.x + kdX * s.xDot;
        s.force = Math.max(-forceMag, Math.min(forceMag, actionForce));
      }

      // Lagrangian physics integration
      const costh = Math.cos(s.theta);
      const sinth = Math.sin(s.theta);
      const totalMass = massCart + massPole;
      const poleMassLength = massPole * length;

      const temp = (s.force + poleMassLength * s.thetaDot * s.thetaDot * sinth) / totalMass;
      const thetaAcc =
        (gravity * sinth - costh * temp) /
        (length * (4.0 / 3.0 - (massPole * costh * costh) / totalMass));
      const xAcc = temp - (poleMassLength * thetaAcc * costh) / totalMass;

      // Euler integration
      s.x += s.xDot * dt;
      s.xDot += xAcc * dt;
      s.theta += s.thetaDot * dt;
      s.thetaDot += thetaAcc * dt;
      s.stepCount += 1;

      // Fail condition: |x| > 2.4m or |theta| > 15 degrees (~0.26 rad)
      const maxAngle = (15 * Math.PI) / 180;
      let statusText = 'Balancing (Normal)';
      if (Math.abs(s.theta) > maxAngle) {
        statusText = 'Fallen (Angle Limit Exceeded)';
      } else if (Math.abs(s.x) > 2.4) {
        statusText = 'Off-track (Position Exceeded)';
      }

      setMetrics({
        x: Number(s.x.toFixed(3)),
        thetaDeg: Number(((s.theta * 180) / Math.PI).toFixed(2)),
        steps: s.stepCount,
        status: statusText,
      });
    };

    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      // Clear background
      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, w, h);

      // Draw horizontal track rail
      const railY = h * 0.7;
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(30, railY);
      ctx.lineTo(w - 30, railY);
      ctx.stroke();

      // Draw grid tick marks
      ctx.fillStyle = '#475569';
      for (let i = -2; i <= 2; i += 0.5) {
        const tickX = w / 2 + (i / 2.4) * (w / 2 - 50);
        ctx.fillRect(tickX - 1, railY - 6, 2, 12);
      }

      // Compute cart screen coordinates
      const s = stateRef.current;
      const cartX = w / 2 + (s.x / 2.4) * (w / 2 - 50);
      const cartW = 70;
      const cartH = 36;
      const cartY = railY - cartH / 2;

      // Draw Cart shadow & body
      ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.fillRect(cartX - cartW / 2 - 4, cartY - 4, cartW + 8, cartH + 8);

      ctx.fillStyle = '#1e293b';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.fillRect(cartX - cartW / 2, cartY, cartW, cartH);
      ctx.strokeRect(cartX - cartW / 2, cartY, cartW, cartH);

      // Draw Cart wheels
      ctx.fillStyle = '#0ea5e9';
      ctx.beginPath();
      ctx.arc(cartX - 20, cartY + cartH, 6, 0, Math.PI * 2);
      ctx.arc(cartX + 20, cartY + cartH, 6, 0, Math.PI * 2);
      ctx.fill();

      // Draw Pole pivot
      const pivotX = cartX;
      const pivotY = cartY;
      ctx.fillStyle = '#f8fafc';
      ctx.beginPath();
      ctx.arc(pivotX, pivotY, 4, 0, Math.PI * 2);
      ctx.fill();

      // Draw Pole
      const poleScreenLength = 120 * (length / 0.5);
      const poleTipX = pivotX + poleScreenLength * Math.sin(s.theta);
      const poleTipY = pivotY - poleScreenLength * Math.cos(s.theta);

      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(pivotX, pivotY);
      ctx.lineTo(poleTipX, poleTipY);
      ctx.stroke();

      // Draw Pole Tip Mass
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(poleTipX, poleTipY, 8 * (massPole / 0.1), 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Draw Force Vector Indicator
      if (Math.abs(s.force) > 0.5) {
        ctx.strokeStyle = '#38bdf8';
        ctx.fillStyle = '#38bdf8';
        ctx.lineWidth = 3;
        const arrowLen = (s.force / forceMag) * 40;
        ctx.beginPath();
        ctx.moveTo(cartX, cartY + cartH / 2);
        ctx.lineTo(cartX + arrowLen, cartY + cartH / 2);
        ctx.stroke();
      }
    };

    const loop = (now: number) => {
      if (running) {
        const elapsed = (now - lastTime) / 1000;
        if (elapsed > 0.015) {
          updatePhysics();
          lastTime = now;
        }
      }
      draw();
      animationFrameId = requestAnimationFrame(loop);
    };

    animationFrameId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [running, autoBalance, gravity, massCart, massPole, length, forceMag]);

  return (
    <div className="space-y-4">
      {/* Simulation Screen */}
      <div className="relative overflow-hidden rounded-xl border border-border bg-slate-950 shadow-inner">
        <canvas
          ref={canvasRef}
          width={640}
          height={280}
          className="w-full h-[240px] md:h-[280px] block"
        />

        {/* Live HUD Overlay */}
        <div className="absolute top-3 left-3 flex flex-wrap items-center gap-2 text-xs">
          <Badge variant={metrics.status.startsWith('Fallen') ? 'destructive' : 'default'}>
            {metrics.status}
          </Badge>
          <Badge variant="secondary" className="font-mono">
            x: {metrics.x} m
          </Badge>
          <Badge variant="secondary" className="font-mono">
            θ: {metrics.thetaDeg}°
          </Badge>
          <Badge variant="secondary" className="font-mono">
            Steps: {metrics.steps}
          </Badge>
        </div>

        {/* Manual force push buttons */}
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              stateRef.current.force = -forceMag;
              stateRef.current.xDot -= 0.5;
            }}
            className="rounded-md bg-slate-900/90 backdrop-blur px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 hover:text-white border border-slate-700/60 shadow-xs cursor-pointer transition-colors"
          >
            ← 向左推 (-F)
          </button>
          <button
            type="button"
            onClick={() => {
              stateRef.current.force = forceMag;
              stateRef.current.xDot += 0.5;
            }}
            className="rounded-md bg-slate-900/90 backdrop-blur px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 hover:text-white border border-slate-700/60 shadow-xs cursor-pointer transition-colors"
          >
            向右推 (+F) →
          </button>
        </div>
      </div>

      {/* Control Buttons Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={() => setRunning(!running)}
            variant={running ? 'secondary' : 'default'}
          >
            {running ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            <span>{running ? '暫停物理引擎' : '開始連續模擬'}</span>
          </Button>

          <Button size="sm" variant="outline" onClick={resetSimulation}>
            <RotateCcw className="h-3.5 w-3.5" />
            <span>重置狀態</span>
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={autoBalance ? 'secondary' : 'outline'}
            onClick={() => setAutoBalance(!autoBalance)}
            className="text-xs"
          >
            <Zap className={`h-3.5 w-3.5 ${autoBalance ? 'text-amber-500' : ''}`} />
            <span>自動 PD 平衡策略: {autoBalance ? '啟用' : '關閉 (純手動)'}</span>
          </Button>
        </div>
      </div>

      {/* Parameter Sliders Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div className="space-y-1.5 rounded-lg border border-border bg-card/60 p-3">
          <div className="flex justify-between font-medium text-foreground">
            <span>小車質量 (Mass Cart)</span>
            <span className="font-mono text-primary font-semibold">{massCart} kg</span>
          </div>
          <Slider
            min={0.2}
            max={3.0}
            step={0.1}
            value={[massCart]}
            onValueChange={([v]) => setMassCart(v)}
          />
        </div>

        <div className="space-y-1.5 rounded-lg border border-border bg-card/60 p-3">
          <div className="flex justify-between font-medium text-foreground">
            <span>擺桿質量 (Mass Pole)</span>
            <span className="font-mono text-primary font-semibold">{massPole} kg</span>
          </div>
          <Slider
            min={0.02}
            max={0.5}
            step={0.02}
            value={[massPole]}
            onValueChange={([v]) => setMassPole(v)}
          />
        </div>

        <div className="space-y-1.5 rounded-lg border border-border bg-card/60 p-3">
          <div className="flex justify-between font-medium text-foreground">
            <span>擺桿半長 (Half Length)</span>
            <span className="font-mono text-primary font-semibold">{length} m</span>
          </div>
          <Slider
            min={0.2}
            max={1.2}
            step={0.05}
            value={[length]}
            onValueChange={([v]) => setLength(v)}
          />
        </div>

        <div className="space-y-1.5 rounded-lg border border-border bg-card/60 p-3">
          <div className="flex justify-between font-medium text-foreground">
            <span>外力極限 (Max Force)</span>
            <span className="font-mono text-primary font-semibold">{forceMag} N</span>
          </div>
          <Slider
            min={2}
            max={30}
            step={1}
            value={[forceMag]}
            onValueChange={([v]) => setForceMag(v)}
          />
        </div>
      </div>
    </div>
  );
};

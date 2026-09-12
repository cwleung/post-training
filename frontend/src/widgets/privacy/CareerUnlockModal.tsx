import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, ShieldCheck, Lock, Unlock, X, KeyRound } from 'lucide-react';
import { usePrivacyStore } from '@/entities/chapter/privacyStore';

export const CareerUnlockModal: React.FC = () => {
  const { isUnlocked, isUnlockModalOpen, closeUnlockModal, unlock, lock } = usePrivacyStore();
  const [passphrase, setPassphrase] = useState('');
  const [error, setError] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);

  useEffect(() => {
    if (isUnlockModalOpen) {
      setPassphrase('');
      setError(false);
    }
  }, [isUnlockModalOpen]);

  // ESC key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isUnlockModalOpen) {
        closeUnlockModal();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isUnlockModalOpen, closeUnlockModal]);

  if (!isUnlockModalOpen) return null;

  const handleUnlock = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passphrase.trim()) return;

    setIsVerifying(true);
    setError(false);
    const success = await unlock(passphrase);
    setIsVerifying(false);

    if (success) {
      setPassphrase('');
      closeUnlockModal();
    } else {
      setError(true);
    }
  };

  const handleLock = () => {
    lock();
    closeUnlockModal();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-md rounded-xl border border-border/80 bg-card p-6 shadow-2xl shadow-black/40 text-card-foreground"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={closeUnlockModal}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
          title="Close (Esc)"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-5">
          <div
            className={`p-2.5 rounded-xl ${
              isUnlocked
                ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30'
                : 'bg-primary/15 text-primary ring-1 ring-primary/30'
            }`}
          >
            {isUnlocked ? <ShieldCheck className="w-5 h-5" /> : <Shield className="w-5 h-5" />}
          </div>
          <div>
            <h3 className="font-title text-base font-bold text-foreground tracking-tight">
              Developer Mode / Advanced Settings
            </h3>
            <p className="text-xs text-muted-foreground">
              {isUnlocked
                ? 'Developer Mode is active. Advanced diagnostics unlocked.'
                : 'Standard Workspace Mode (Enter developer key for advanced diagnostics)'}
            </p>
          </div>
        </div>

        {/* Unlocked State View */}
        {isUnlocked ? (
          <div className="space-y-4">
            <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3.5 text-xs text-emerald-300 flex items-start gap-2.5">
              <Unlock className="w-4 h-4 mt-0.5 shrink-0 text-emerald-400" />
              <div>
                <p className="font-semibold text-emerald-200">Developer Mode Active</p>
                <p className="text-emerald-300/80 mt-0.5 leading-relaxed">
                  Advanced system diagnostics, architecture blueprints, and deep-dive workbenches are active on this browser.
                </p>
              </div>
            </div>

            <div className="flex gap-3 pt-2">
              <button
                type="button"
                onClick={handleLock}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25 transition-all shadow-sm shadow-rose-500/10"
              >
                <Lock className="w-3.5 h-3.5" />
                Disable Developer Mode
              </button>
              <button
                type="button"
                onClick={closeUnlockModal}
                className="px-4 py-2.5 rounded-lg text-xs font-medium bg-muted/60 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        ) : (
          /* Locked State Form */
          <form onSubmit={handleUnlock} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-foreground/80 mb-1.5 flex items-center gap-1.5">
                <KeyRound className="w-3.5 h-3.5 text-muted-foreground" />
                Developer Key
              </label>
              <input
                type="password"
                value={passphrase}
                onChange={(e) => setPassphrase(e.target.value)}
                placeholder="Enter developer access key..."
                autoFocus
                className="w-full px-3.5 py-2.5 text-sm bg-background border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary text-foreground placeholder:text-muted-foreground/60 transition-all font-mono"
              />
              {error && (
                <div className="flex items-center gap-1.5 mt-2 text-xs text-rose-400">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  <span>Invalid developer key. Access denied.</span>
                </div>
              )}
            </div>

            <div className="flex gap-3 pt-1">
              <button
                type="submit"
                disabled={isVerifying || !passphrase.trim()}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-50 transition-all shadow-md shadow-primary/20"
              >
                <Unlock className="w-3.5 h-3.5" />
                {isVerifying ? 'Authenticating...' : 'Enable Developer Mode'}
              </button>
              <button
                type="button"
                onClick={closeUnlockModal}
                className="px-4 py-2.5 rounded-lg text-xs font-medium bg-muted/60 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

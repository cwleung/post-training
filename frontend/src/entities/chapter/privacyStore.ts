import { create } from 'zustand';

// SHA-256 hash of 'frontier-mle-2026'
const TARGET_SHA256_HASH = '2c9da6721d235446090e2b4276993e0c4a66a7b384d5004e8a32e66b0bbb6c75';
const STORAGE_KEY = 'deepagents_career_unlocked';

interface PrivacyState {
  isUnlocked: boolean;
  isUnlockModalOpen: boolean;
  openUnlockModal: () => void;
  closeUnlockModal: () => void;
  unlock: (passphrase: string) => Promise<boolean>;
  lock: () => void;
}

async function computeSha256(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

export const usePrivacyStore = create<PrivacyState>((set) => ({
  isUnlocked: typeof window !== 'undefined' && localStorage.getItem(STORAGE_KEY) === 'true',
  isUnlockModalOpen: false,

  openUnlockModal: () => set({ isUnlockModalOpen: true }),
  closeUnlockModal: () => set({ isUnlockModalOpen: false }),

  unlock: async (passphrase: string) => {
    try {
      const computedHash = await computeSha256(passphrase.trim());
      if (computedHash === TARGET_SHA256_HASH) {
        if (typeof window !== 'undefined') {
          localStorage.setItem(STORAGE_KEY, 'true');
        }
        set({ isUnlocked: true });
        return true;
      }
      return false;
    } catch (err) {
      console.error('Error verifying passphrase:', err);
      return false;
    }
  },

  lock: () => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
    set({ isUnlocked: false });
  },
}));

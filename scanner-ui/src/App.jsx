import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Background from './components/Background';
import GlassCard from './components/GlassCard';
import Scanner from './components/Scanner';
import ProcessingLoader from './components/ProcessingLoader';
import { ShieldCheck, RefreshCcw } from 'lucide-react';

function App() {
  const [appState, setAppState] = useState('IDLE'); // IDLE, PROCESSING, RESULT
  const [scannedImage, setScannedImage] = useState(null);

  const handleScanStart = (file) => {
    setScannedImage(URL.createObjectURL(file));
    setAppState('PROCESSING');
  };

  const handleProcessingComplete = () => {
    setAppState('RESULT');
  };

  const resetScan = () => {
    setAppState('IDLE');
    setScannedImage(null);
  };

  return (
    <div className="min-h-screen text-white font-sans selection:bg-cyan-500/30">
      <Background />

      <main className="container mx-auto px-4 min-h-screen flex flex-col items-center justify-center relative z-10">

        {/* Header HUD */}
        <motion.header
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="absolute top-0 left-0 right-0 p-6 flex justify-between items-center pointer-events-none"
        >
          <div className="flex items-center space-x-2">
            <ShieldCheck className="text-cyan-400" size={24} />
            <span className="text-cyan-400 font-bold tracking-widest text-sm">BIO-SAFE SYSTEMS v2.0</span>
          </div>
          <div className="flex items-center space-x-4 text-xs text-slate-400 font-mono">
            <span>SYS: ONLINE</span>
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          </div>
        </motion.header>

        <AnimatePresence mode="wait">
          {appState === 'IDLE' && (
            <motion.div
              key="scanner"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
              className="w-full max-w-md"
            >
              <GlassCard className="p-1">
                <div className="text-center pt-8 pb-2">
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-cyan-200 bg-clip-text text-transparent mb-2">
                    Biometric Identity
                  </h1>
                  <p className="text-slate-400 text-sm">Place finger on sensor to begin analysis</p>
                </div>
                <Scanner onScanStart={handleScanStart} />
              </GlassCard>
            </motion.div>
          )}

          {appState === 'PROCESSING' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, y: -20, filter: "blur(10px)" }}
              className="w-full max-w-md"
            >
              <GlassCard>
                <ProcessingLoader onComplete={handleProcessingComplete} />
              </GlassCard>
            </motion.div>
          )}

          {appState === 'RESULT' && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full max-w-md text-center"
            >
              <GlassCard className="p-8 space-y-6">
                <div className="space-y-2">
                  <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Blood Group Detected</h2>
                  <div className="relative inline-block">
                    <h1 className="text-8xl font-black text-white drop-shadow-[0_0_20px_rgba(0,243,255,0.5)]">
                      B+
                    </h1>
                    <div className="absolute -inset-4 bg-cyan-500/20 blur-xl rounded-full -z-10" />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between text-sm text-slate-300">
                    <span>Confidence Score</span>
                    <span className="text-cyan-400 font-bold">98.4%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-cyan-400 shadow-[0_0_10px_#00f3ff]"
                      initial={{ width: 0 }}
                      animate={{ width: "98.4%" }}
                      transition={{ duration: 1, delay: 0.5 }}
                    />
                  </div>
                </div>

                <button
                  onClick={resetScan}
                  className="w-full py-4 mt-4 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/50 rounded-xl text-cyan-400 font-bold uppercase tracking-wider transition-all hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] flex items-center justify-center space-x-2"
                >
                  <RefreshCcw size={18} />
                  <span>Scan New Sample</span>
                </button>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}

export default App;

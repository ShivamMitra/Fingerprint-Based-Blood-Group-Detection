import { motion } from 'framer-motion';
import { Scan, Upload, Fingerprint } from 'lucide-react';
import { useState, useRef } from 'react';

export default function Scanner({ onScanStart }) {
    const [isHovering, setIsHovering] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            onScanStart(file);
        }
    };

    return (
        <div
            className="flex flex-col items-center justify-center p-8 pb-10 cursor-pointer group"
            onClick={() => fileInputRef.current?.click()}
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
        >
            {/* Scanner Area */}
            <div className="relative mb-8">
                {/* Outer Ring */}
                <div className="w-64 h-64 rounded-full border-2 border-cyan-500/30 flex items-center justify-center relative overflow-hidden bg-slate-900/50 backdrop-blur-sm transition-all duration-300 group-hover:border-cyan-400 group-hover:shadow-[0_0_30px_rgba(0,243,255,0.2)]">

                    {/* Scanning Animation Layer */}
                    <div className="absolute inset-0 z-0">
                        <div className="w-full h-[2px] bg-cyan-400 shadow-[0_0_15px_#00f3ff] absolute top-0 animate-[scan_2s_ease-in-out_infinite]" />
                    </div>

                    {/* Fingerprint Icon */}
                    <Fingerprint
                        className={`w-32 h-32 text-cyan-500/80 transition-all duration-500 ${isHovering ? 'scale-110 opacity-100' : 'scale-100 opacity-60'}`}
                        strokeWidth={1}
                    />

                    {/* Decorative HUD Elements */}
                    <div className="absolute inset-0 rounded-full border border-cyan-500/10 scale-90" />
                    <div className="absolute inset-0 rounded-full border border-cyan-500/10 scale-75" />

                    {/* Rotating Bracket Accents */}
                    <motion.div
                        className="absolute inset-0 border-t-2 border-cyan-400 rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                    />
                    <motion.div
                        className="absolute inset-0 border-b-2 border-cyan-400/50 rounded-full scale-110"
                        animate={{ rotate: -360 }}
                        transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
                    />
                </div>
            </div>

            {/* Hidden Input */}
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept="image/*"
            />

            {/* Upload Label — in normal flow, not absolute */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center space-x-2 text-cyan-400 text-sm tracking-wider uppercase font-medium bg-cyan-950/30 px-6 py-3 rounded-full border border-cyan-500/20 backdrop-blur-md transition-all duration-300 group-hover:bg-cyan-500/10 group-hover:shadow-[0_0_20px_rgba(0,243,255,0.15)]"
            >
                <Upload size={14} />
                <span>Click to Scan Fingerprint</span>
            </motion.div>
        </div>
    );
}

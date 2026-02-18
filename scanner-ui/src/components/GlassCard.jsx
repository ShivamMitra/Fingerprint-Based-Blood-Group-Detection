import { motion } from 'framer-motion';

export default function GlassCard({ children, className = "", delay = 0 }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay, ease: "easeOut" }}
            className={`relative overflow-hidden rounded-2xl border border-cyan-500/20 bg-slate-900/40 backdrop-blur-xl shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] ${className}`}
        >
            {/* Glossy Reflection */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />

            {/* Content */}
            <div className="relative z-10">
                {children}
            </div>
        </motion.div>
    );
}

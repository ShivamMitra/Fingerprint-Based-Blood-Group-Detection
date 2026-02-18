import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { Cpu, Activity, Search, ShieldCheck } from 'lucide-react';

const steps = [
    { text: "Initializing Neural Network...", icon: Cpu },
    { text: "Extracting Ridge Features...", icon: Search },
    { text: "Analyzing Minutiae Points...", icon: Activity },
    { text: "Finalizing Classification...", icon: ShieldCheck },
];

export default function ProcessingLoader({ onComplete }) {
    const [currentStep, setCurrentStep] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentStep((prev) => {
                if (prev < steps.length - 1) {
                    return prev + 1;
                }
                clearInterval(interval);
                setTimeout(onComplete, 800);
                return prev;
            });
        }, 1500);

        return () => clearInterval(interval);
    }, [onComplete]);

    const CurrentIcon = steps[currentStep].icon;

    return (
        <div className="flex flex-col items-center justify-center p-12 space-y-8">
            {/* DNA/Loader Animation */}
            <div className="relative w-40 h-40">
                <motion.div
                    className="absolute inset-0 rounded-full border-4 border-cyan-500/20 border-t-cyan-400"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                />
                <motion.div
                    className="absolute inset-4 rounded-full border-4 border-blue-500/20 border-b-blue-400"
                    animate={{ rotate: -360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                        key={currentStep}
                        initial={{ scale: 0.5, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.5, opacity: 0 }}
                    >
                        <CurrentIcon className="w-12 h-12 text-cyan-400" />
                    </motion.div>
                </div>
            </div>

            {/* Text Feedback */}
            <div className="text-center space-y-2">
                <motion.h3
                    key={steps[currentStep].text}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-xl font-bold text-cyan-400 tracking-wider"
                >
                    {steps[currentStep].text}
                </motion.h3>
                <div className="w-64 h-1 bg-slate-800 rounded-full overflow-hidden mt-4">
                    <motion.div
                        className="h-full bg-gradient-to-r from-blue-500 to-cyan-400"
                        initial={{ width: "0%" }}
                        animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                        transition={{ duration: 0.5 }}
                    />
                </div>
            </div>
        </div>
    );
}

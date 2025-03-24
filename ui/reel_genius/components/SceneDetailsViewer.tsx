"use client"
import {useState} from 'react';
import Image from 'next/image';
import {motion, AnimatePresence} from 'framer-motion';
import {SceneMediaInfo} from '@/lib/api';
import {fadeIn, slideUp, cardReveal} from './animations';

interface SceneDetailsViewerProps {
    scenes: SceneMediaInfo[];
    showPrompts?: boolean;
}

export default function SceneDetailsViewer({scenes, showPrompts = false}: SceneDetailsViewerProps) {
    const [selectedSceneIndex, setSelectedSceneIndex] = useState<number | null>(0);

    // If no scenes, show a message
    if (!scenes || scenes.length === 0) {
        return (
            <div className="p-4 text-center text-muted-foreground">
                No scene data available
            </div>
        );
    }

    const selectedScene = selectedSceneIndex !== null ? scenes[selectedSceneIndex] : null;

    return (
        <motion.div
            className="flex flex-col space-y-4"
            initial="hidden"
            animate="visible"
            variants={fadeIn}
        >
            {/* Scene navigation */}
            <motion.div
                className="flex flex-wrap gap-2"
                variants={fadeIn}
            >
                {scenes.map((scene, index) => (
                    <motion.button
                        key={index}
                        onClick={() => setSelectedSceneIndex(index)}
                        className={`px-3 py-1 text-xs rounded-full ${
                            selectedSceneIndex === index
                                ? 'bg-primary text-primary-foreground'
                                : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                        }`}
                        whileHover={{
                            scale: 1.05,
                            backgroundColor: selectedSceneIndex === index ? undefined : 'rgba(0,0,0,0.1)'
                        }}
                        whileTap={{scale: 0.95}}
                        initial={{opacity: 0, y: 10}}
                        animate={{opacity: 1, y: 0}}
                        transition={{
                            duration: 0.2,
                            delay: index * 0.05
                        }}
                    >
                        Scene {index + 1}
                    </motion.button>
                ))}
            </motion.div>

            {/* Selected scene details */}
            <AnimatePresence mode="wait">
                {selectedScene && (
                    <motion.div
                        key={selectedSceneIndex}
                        className="grid grid-cols-1 md:grid-cols-2 gap-6"
                        variants={cardReveal}
                        initial="hidden"
                        animate="visible"
                        exit={{opacity: 0, y: 10, transition: {duration: 0.2}}}
                        layout
                    >
                        {/* Scene media */}
                        <motion.div
                            className="space-y-4"
                            variants={slideUp}
                        >
                            {/* Image */}
                            {selectedScene.image_url && (
                                <motion.div
                                    className="rounded-md overflow-hidden border shadow-sm"
                                    initial={{opacity: 0, scale: 0.9}}
                                    animate={{opacity: 1, scale: 1}}
                                    transition={{duration: 0.4}}
                                    whileHover={{scale: 1.02}}
                                >
                                    <div className="relative w-full h-64">
                                        <Image
                                            src={selectedScene.image_url}
                                            alt={`Scene ${selectedSceneIndex! + 1} Image`}
                                            fill
                                            className="object-contain"
                                        />
                                    </div>
                                </motion.div>
                            )}

                            {/* Audio */}
                            {selectedScene.audio_url && (
                                <motion.div
                                    className="p-3 bg-muted/40 rounded-md"
                                    variants={fadeIn}
                                    initial="hidden"
                                    animate="visible"
                                    transition={{delay: 0.2}}
                                >
                                    <h4 className="text-xs text-muted-foreground mb-2">Scene Audio</h4>
                                    <audio src={selectedScene.audio_url} controls className="w-full max-w-full"/>
                                </motion.div>
                            )}
                        </motion.div>

                        {/* Scene text and details */}
                        <motion.div
                            className="space-y-4"
                            variants={slideUp}
                        >
                            {/* Text content */}
                            <motion.div
                                className="p-4 bg-muted/40 rounded-md"
                                initial={{opacity: 0, y: 10}}
                                animate={{opacity: 1, y: 0}}
                                transition={{duration: 0.3}}
                            >
                                <h3 className="font-medium mb-2">Scene Text</h3>
                                <p className="text-sm whitespace-pre-line">{selectedScene.text || 'No text available'}</p>
                            </motion.div>

                            {/* Scene details */}
                            <motion.div
                                className="grid grid-cols-2 gap-3 text-sm"
                                variants={fadeIn}
                                transition={{delay: 0.1}}
                            >
                                <div>
                                    <h4 className="text-xs text-muted-foreground">Duration</h4>
                                    <p>{selectedScene.duration?.toFixed(1) || 'N/A'} seconds</p>
                                </div>

                                <div>
                                    <h4 className="text-xs text-muted-foreground">Transition</h4>
                                    <p className="capitalize">{selectedScene.transition || 'Default'}</p>
                                </div>

                                {selectedScene.image_size_bytes && (
                                    <div>
                                        <h4 className="text-xs text-muted-foreground">Image Size</h4>
                                        <p>{formatFileSize(selectedScene.image_size_bytes)}</p>
                                    </div>
                                )}

                                {selectedScene.audio_size_bytes && (
                                    <div>
                                        <h4 className="text-xs text-muted-foreground">Audio Size</h4>
                                        <p>{formatFileSize(selectedScene.audio_size_bytes)}</p>
                                    </div>
                                )}
                            </motion.div>

                            {/* Prompt information - only shown if requested */}
                            {showPrompts && (
                                <motion.div
                                    className="space-y-3"
                                    initial={{opacity: 0}}
                                    animate={{opacity: 1}}
                                    transition={{delay: 0.3}}
                                >
                                    {selectedScene.image_prompt && (
                                        <motion.div
                                            className="p-3 bg-muted/20 rounded-md"
                                            whileHover={{backgroundColor: 'rgba(0,0,0,0.1)'}}
                                        >
                                            <h4 className="text-xs text-muted-foreground mb-1">Image Prompt</h4>
                                            <p className="text-xs font-mono whitespace-pre-line">{selectedScene.image_prompt}</p>
                                        </motion.div>
                                    )}

                                    {selectedScene.visual_description && (
                                        <motion.div
                                            className="p-3 bg-muted/20 rounded-md"
                                            whileHover={{backgroundColor: 'rgba(0,0,0,0.1)'}}
                                        >
                                            <h4 className="text-xs text-muted-foreground mb-1">Visual Description</h4>
                                            <p className="text-xs whitespace-pre-line">{selectedScene.visual_description}</p>
                                        </motion.div>
                                    )}
                                </motion.div>
                            )}
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}

// Helper function to format file size in bytes to a human-readable format
function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
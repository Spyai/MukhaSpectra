import type { Transition } from 'motion/react';

/** Purposeful, spring-based motion (150–400 ms). MotionConfig reducedMotion="user" disables
 * transforms for people who ask for less motion; opacity fades remain. */
export const spring: Transition = { type: 'spring', stiffness: 380, damping: 34, mass: 0.9 };
export const softSpring: Transition = { type: 'spring', stiffness: 220, damping: 30 };
export const fade: Transition = { duration: 0.22, ease: [0.22, 1, 0.36, 1] };

export const staggerContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
};

export const riseItem = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: softSpring },
};

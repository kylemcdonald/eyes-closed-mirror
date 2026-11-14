// Shared MediaPipe helpers + landmark sets for the eye-region swap.
export const MP_VERSION = '0.10.14';
const MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export async function createLandmarker(runningMode = 'VIDEO') {
  const { FaceLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const make = (delegate) => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode, numFaces: 1, outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await make('GPU'); } catch { return await make('CPU'); }
}

// Landmarks that DON'T move when the eyelids close. The eyelid contour itself (and the upper-lid crease)
// is deliberately excluded, so triangles span across the eye and the closed lid in the source image is
// carried over rigidly instead of being stretched to match the open lid in the live frame.
// CORE vertices get blend weight 1, OUTER vertices weight 0: the blend fades out between them.
const R = {
  corners: [33, 133],
  ring2lower: [130, 110, 24, 23, 22, 26, 112, 243],
  ring3: [226, 31, 228, 229, 230, 231, 232, 233, 244, 189, 221, 222, 223, 224, 225, 113],
  browLower: [46, 53, 52, 65, 55],
  browUpper: [70, 63, 105, 66, 107],
  ring4: [143, 111, 117, 118, 119, 120, 121, 128, 245],
};
const L = {
  corners: [263, 362],
  ring2lower: [359, 339, 254, 253, 252, 256, 341, 463],
  ring3: [446, 261, 448, 449, 450, 451, 452, 453, 464, 413, 441, 442, 443, 444, 445, 342],
  browLower: [276, 283, 282, 295, 285],
  browUpper: [300, 293, 334, 296, 336],
  ring4: [372, 340, 346, 347, 348, 349, 350, 357, 465],
};
export const CORE = [...R.corners, ...R.ring2lower, ...R.ring3, ...R.browLower,
  ...L.corners, ...L.ring2lower, ...L.ring3, ...L.browLower, 168, 6];
export const OUTER = [...R.browUpper, ...R.ring4, ...L.browUpper, ...L.ring4, 9, 197, 139, 368];
export const STABLE = [...CORE, ...OUTER];
export const WEIGHTS = STABLE.map((_, i) => (i < CORE.length ? 1 : 0));

export const EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

export function blendshape(result, name) {
  const cats = result?.faceBlendshapes?.[0]?.categories;
  return cats?.find((c) => c.categoryName === name)?.score ?? null;
}
export const eyesClosedScore = (r) => {
  const a = blendshape(r, 'eyeBlinkLeft'), b = blendshape(r, 'eyeBlinkRight');
  return a == null ? null : (a + b) / 2;
};

// Mean displacement of the stable landmarks between two detections, in units of inter-ocular distance.
// Small = the edit preserved the face geometry, so it can be mapped back onto the original capture.
export function stableDrift(a, b) {
  const iod = Math.hypot(a[33].x - a[263].x, a[33].y - a[263].y);
  let s = 0;
  for (const i of STABLE) s += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
  return s / STABLE.length / iod;
}

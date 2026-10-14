/**
 * Home hero (lazy chunk): the principle, literally. A softly undulating tissue surface; a 405 nm
 * violet light cone sweeps across it; tissue behind the beam fluoresces green and slowly fades,
 * except one patch that stays dark (fluorescence visualisation loss) — ringed as it is revealed.
 */
import { Canvas, useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

const SPAN = 6.0; // beam travels from -SPAN/2 to +SPAN/2 (beyond the visible surface)
const PERIOD = 7.5; // seconds per sweep

const vertex = /* glsl */ `
  uniform float uTime;
  varying vec3 vWorld;
  varying vec2 vXZ;
  float h(vec2 p, float t) {
    return 0.07 * sin(p.x * 1.7 + t * 0.55) * cos(p.y * 2.3 - t * 0.35)
         + 0.035 * sin(p.x * 4.1 - p.y * 3.3 + t * 0.8)
         + 0.012 * sin(p.x * 11.0 + p.y * 9.0);
  }
  void main() {
    vec3 p = position;
    // plane is XY before rotation; we displace along +Z (becomes +Y after rotation)
    vec2 xz = vec2(p.x, -p.y);
    p.z += h(xz, uTime);
    vXZ = xz;
    vec4 world = modelMatrix * vec4(p, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fragment = /* glsl */ `
  uniform float uTime;
  uniform float uBeamX;
  uniform float uBeamZ;
  uniform vec2 uFvl;
  varying vec3 vWorld;
  varying vec2 vXZ;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
  }

  void main() {
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (n.y < 0.0) n = -n;
    vec3 L = normalize(vec3(uBeamX - vWorld.x, 2.4, uBeamZ - vWorld.z));
    float diffuse = clamp(dot(n, L), 0.0, 1.0);

    // tissue texture
    float tex = 0.75 + 0.25 * noise(vXZ * 9.0) + 0.1 * noise(vXZ * 31.0);

    // FVL patch: irregular blob
    vec2 d = vXZ - uFvl;
    float ang = atan(d.y, d.x);
    float r = 0.36 + 0.05 * sin(ang * 3.0 + 1.3) + 0.035 * sin(ang * 7.0) + 0.04 * (noise(vXZ * 6.0) - 0.5);
    float dist = length(d);
    float fvl = smoothstep(r + 0.05, r - 0.06, dist);

    // beam footprint (ellipse) and afterglow trail behind it
    float bx = (vXZ.x - uBeamX) / 0.42;
    float bz = (vXZ.y - uBeamZ) / 1.55;
    float footprint = exp(-(bx * bx + bz * bz) * 1.6);
    float behind = uBeamX - vXZ.x;
    float age = behind >= 0.0 ? behind : behind + ${SPAN.toFixed(2)};
    float zcover = smoothstep(2.2, 1.2, abs(vXZ.y - uBeamZ));
    float glow = exp(-age / 1.35) * smoothstep(4.4, 2.6, age) * zcover;
    glow = max(glow, footprint);

    // Colours are authored in display (sRGB) space; no colour-space conversion below.
    vec3 ink = vec3(0.027, 0.035, 0.06);
    vec3 mucosa = vec3(0.30, 0.13, 0.19) * tex;
    vec3 col = mucosa * (0.25 + 0.75 * diffuse) * 0.55;

    // violet excitation reflected where the beam lands
    col += vec3(0.36, 0.2, 0.95) * footprint * (0.35 + 0.3 * diffuse);

    // green autofluorescence, lost inside the FVL patch (which stays dark)
    float fl = glow * (1.0 - 0.92 * fvl) * tex;
    col = mix(col, col * 0.35, fvl * glow);
    col += vec3(0.12, 0.85, 0.36) * fl;
    col += vec3(0.1, 0.35, 0.12) * fl * fl;

    // "look closer" dashed ring revealed by the passing glow
    float ringR = r + 0.12;
    float ring = smoothstep(0.02, 0.0, abs(dist - ringR));
    float dash = step(0.45, fract(ang * 10.0 / 6.28318));
    float reveal = smoothstep(0.08, 0.45, exp(-max(uBeamX - uFvl.x, 0.0) / 1.6) * step(uFvl.x - 0.2, uBeamX));
    col += vec3(0.95) * ring * dash * reveal * 0.8;

    // fade the surface edges into the background ink
    float edge = smoothstep(2.7, 1.9, abs(vXZ.x)) * smoothstep(1.95, 1.1, abs(vXZ.y));
    col = mix(ink, col, edge);
    gl_FragColor = vec4(col, 1.0);
  }
`;

function Tissue({ reduced }: { reduced: boolean }) {
  const mat = useRef<THREE.ShaderMaterial>(null);
  const cone = useRef<THREE.Group>(null);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uBeamX: { value: -0.4 },
      uBeamZ: { value: 0 },
      uFvl: { value: new THREE.Vector2(0.55, 0.12) },
    }),
    [],
  );

  useFrame((state) => {
    const t = reduced ? 3.1 : state.clock.elapsedTime;
    const beamX = -SPAN / 2 + ((t % PERIOD) / PERIOD) * SPAN;
    const beamZ = 0.18 * Math.sin(t * 0.5);
    // R3F copies `uniforms` into the material's own objects, so update them through the ref.
    const u = mat.current?.uniforms;
    if (u) {
      u.uTime!.value = reduced ? 0 : t;
      u.uBeamX!.value = beamX;
      u.uBeamZ!.value = beamZ;
    }
    if (cone.current) cone.current.position.set(beamX, 0, beamZ);
  });

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[5.8, 4.2, 300, 220]} />
        <shaderMaterial
          ref={mat}
          vertexShader={vertex}
          fragmentShader={fragment}
          uniforms={uniforms}
        />
      </mesh>
      <group ref={cone}>
        {/* light cone: apex at the LED, base on the tissue */}
        <mesh position={[0, 0.95, 0]}>
          <coneGeometry args={[0.5, 1.9, 48, 1, true]} />
          <meshBasicMaterial
            color="#7b5cff"
            transparent
            opacity={0.1}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
        <mesh position={[0, 0.95, 0]} scale={[0.55, 1, 0.55]}>
          <coneGeometry args={[0.5, 1.9, 48, 1, true]} />
          <meshBasicMaterial
            color="#9d86ff"
            transparent
            opacity={0.12}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
        <mesh position={[0, 1.9, 0]}>
          <sphereGeometry args={[0.05, 24, 24]} />
          <meshBasicMaterial color="#c9bbff" />
        </mesh>
      </group>
    </group>
  );
}

export default function HeroScene({ reduced }: { reduced: boolean }) {
  return (
    <Canvas
      camera={{ position: [0, 2.05, 2.3], fov: 38 }}
      dpr={[1, 1.75]}
      frameloop={reduced ? 'demand' : 'always'}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      onCreated={({ camera }) => camera.lookAt(0, -0.05, 0.05)}
    >
      <Tissue reduced={reduced} />
    </Canvas>
  );
}

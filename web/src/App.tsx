import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

type Geom = { id: number; name: string; body: number; body_name: string; type: number; size: number[]; pos: number[]; quat: number[]; rgba: number[]; is_pool: boolean; mesh_id?: number };
type SceneData = { geoms: Geom[]; body_order: number[]; pool: Record<string, {capacity:number; names:string[]; active:string[]}> };
type LiveState = { active_objects:string[]; prompt:string; running:boolean; status:string };

function useSocket(onState: (s: LiveState) => void) {
  const socket = useRef<WebSocket | null>(null);
  const frame = useRef<Float32Array | null>(null);
  const onStateRef = useRef(onState);
  onStateRef.current = onState;
  useEffect(() => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer'; socket.current = ws;
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') { const msg = JSON.parse(event.data) as LiveState & {type:string}; if (msg.type === 'state') onStateRef.current(msg); }
      else frame.current = new Float32Array(event.data as ArrayBuffer);
    };
    return () => { ws.close(); socket.current = null; };
  }, []);
  return { socket, frame };
}

function GeomMesh({ geom, active }: {geom:Geom; active:boolean}) {
  const ref = useRef<THREE.Mesh>(null);
  const material = useMemo(() => new THREE.MeshStandardMaterial({color:new THREE.Color(geom.rgba[0],geom.rgba[1],geom.rgba[2]), transparent:geom.rgba[3]<1, opacity:geom.rgba[3], roughness:.75}), [geom]);
  const [meshGeometry,setMeshGeometry]=useState<THREE.BufferGeometry|null>(null);
  useEffect(() => { if (ref.current) { ref.current.position.fromArray(geom.pos); ref.current.quaternion.set(geom.quat[1],geom.quat[2],geom.quat[3],geom.quat[0]); } }, [geom]);
  const geometry=useMemo(()=>{const size=geom.size;
    if (geom.type===6) return new THREE.BoxGeometry(size[0]*2,size[1]*2,size[2]*2);
    if (geom.type===2) return new THREE.SphereGeometry(size[0],24,16);
    if (geom.type===5) return new THREE.CylinderGeometry(size[0],size[0],size[1]*2,20);
    if (geom.type===4) return new THREE.SphereGeometry(1,20,14).scale(size[0],size[1],size[2]);
    if (geom.type===3) return new THREE.CapsuleGeometry(size[0],size[1]*2,6,12);
    return new THREE.BoxGeometry(Math.max(size[0]*2,.02),Math.max(size[1]*2,.02),Math.max(size[2]*2,.02));
  },[geom]);
  useEffect(()=>{ if(geom.type!==7||geom.mesh_id===undefined)return; let alive=true; fetch(`/api/mesh/${geom.mesh_id}`).then(r=>r.arrayBuffer()).then(bytes=>{if(alive)setMeshGeometry(new STLLoader().parse(bytes));}); return()=>{alive=false;}; },[geom]);
  return <mesh ref={ref} geometry={geom.type===7?(meshGeometry??geometry):geometry} material={material} visible={!geom.is_pool || active} castShadow receiveShadow />;
}

function BodyGroup({bodyId, geoms, frames, activeSlots}:{bodyId:number;geoms:Geom[];frames:React.MutableRefObject<Float32Array|null>;activeSlots:Set<string>}) {
  const bodyRef=useRef<THREE.Group>(null);
  useFrame(()=>{const pose=frames.current; if(!pose||!bodyRef.current)return; const offset=bodyId*7; bodyRef.current.position.set(pose[offset],pose[offset+1],pose[offset+2]); bodyRef.current.quaternion.set(pose[offset+4],pose[offset+5],pose[offset+6],pose[offset+3]);});
  return <group ref={bodyRef}>{geoms.map(g=><GeomMesh key={g.id} geom={g} active={!g.is_pool||activeSlots.has(g.name.replace('_geom',''))}/>)}</group>;
}

function Scene({data, frames, active}:{data:SceneData;frames:React.MutableRefObject<Float32Array|null>;active:string[]}) {
  const activeSlots = useMemo(() => new Set(Object.values(data.pool).flatMap(p=>p.names.filter((_,i)=>p.active.includes(p.names[i])))),[data]);
  const slotsFromState=useMemo(()=>new Set(data.geoms.filter(g=>g.is_pool&&active.includes(g.name.replace('_geom',''))).map(g=>g.name.replace('_geom',''))),[active,data]);
  const visibleSlots=useMemo(()=>new Set([...activeSlots,...slotsFromState]),[activeSlots,slotsFromState]);
  const byBody=useMemo(()=>data.geoms.reduce<Record<number,Geom[]>>((groups,g)=>{(groups[g.body]??=[]).push(g);return groups;},{}),[data]);
  return <>
    {Object.entries(byBody).map(([id,geoms])=><BodyGroup key={id} bodyId={Number(id)} geoms={geoms} frames={frames} activeSlots={visibleSlots}/>)}
    <Grid position={[0,0,.751]} args={[2,2]} cellSize={.1} cellThickness={.5} sectionSize={.5} sectionThickness={1} rotation={[Math.PI/2,0,0]} />
  </>;
}

export function App() {
  const [scene,setScene]=useState<SceneData|null>(null); const [live,setLive]=useState<LiveState>({active_objects:[],prompt:'',running:false,status:'connecting'});
  const [prompt,setPrompt]=useState(''); const [paused,setPaused]=useState(false); const [cameraOpen,setCameraOpen]=useState(true);
  const stableState=(s:LiveState)=>{setLive(s);setPrompt(s.prompt);}; const {socket,frame}=useSocket(stableState);
  useEffect(()=>{fetch('/api/scene').then(r=>r.json()).then(setScene).catch(()=>setLive(s=>({...s,status:'API unavailable'})));},[]);
  const command=(command:string, extra:Record<string,unknown>={})=>socket.current?.readyState===WebSocket.OPEN&&socket.current.send(JSON.stringify({command,...extra}));
  return <main className="app"><Canvas shadows camera={{position:[1.5,-1.8,1.5],fov:45}} onCreated={({camera})=>{camera.up.set(0,0,1);camera.lookAt(0,0,.7);}}>
    <color attach="background" args={['#080c12']} /><ambientLight intensity={1.3}/><directionalLight position={[1,-2,3]} intensity={2.5} castShadow />
    {scene&&<Suspense fallback={null}><Scene data={scene} frames={frame} active={live.active_objects}/></Suspense>}
    <OrbitControls makeDefault target={[0,0,.7]} />
  </Canvas>
  <header className="prompt panel"><span className={`status ${live.running?'on':''}`}/><input value={prompt} onChange={e=>{setPrompt(e.target.value);command('set_prompt',{text:e.target.value});}} placeholder="Describe what the robot should do…"/><button className="primary" onClick={()=>command('run')}>Run</button><button onClick={()=>command('stop')}>Stop</button></header>
  <aside className="toolbar panel"><button onClick={()=>command('reset')}>↺ Reset</button><button onClick={()=>{command(paused?'resume':'pause');setPaused(!paused);}}>{paused?'▶ Resume':'Ⅱ Pause'}</button></aside>
  <section className="tray panel"><div className="tray-title"><b>OBJECTS</b><span>Drag an object onto the table</span></div><div className="cards">{(['box','cylinder','sphere'] as const).map(kind=><div className="card" key={kind}><i className={`shape ${kind}`}/><div><b>{kind}</b><small>{scene?.pool[kind]?.active.length??0}/{scene?.pool[kind]?.capacity??10} in scene</small></div></div>)}</div></section>
  <section className="camera panel"><button className="camera-toggle" onClick={()=>setCameraOpen(!cameraOpen)}>CAMERAS <span>{cameraOpen?'⌄':'⌃'}</span></button>{cameraOpen&&<div className="feeds"><img src="/api/camera/front.jpg"/><img src="/api/camera/top.jpg"/></div>}</section>
  <div className="connection">{live.status}</div></main>;
}

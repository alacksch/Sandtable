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
  const pending = useRef(new Map<string,{resolve:(value:unknown)=>void;reject:(reason:Error)=>void}>());
  onStateRef.current = onState;
  useEffect(() => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer'; socket.current = ws;
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') { const msg = JSON.parse(event.data) as LiveState & {type:string;id?:string;code?:string;message?:string};
        if(msg.type==='ack'&&msg.id){pending.current.get(msg.id)?.resolve(msg);pending.current.delete(msg.id);}
        else if(msg.type==='error'&&msg.id){pending.current.get(msg.id)?.reject(new Error(`${msg.code}: ${msg.message}`));pending.current.delete(msg.id);}
        else if (msg.type === 'state') onStateRef.current(msg);
      }
      else frame.current = new Float32Array(event.data as ArrayBuffer);
    };
    return () => { ws.close(); socket.current = null; };
  }, []);
  const request=(command:string, extra:Record<string,unknown>={})=>new Promise<unknown>((resolve,reject)=>{
    if(socket.current?.readyState!==WebSocket.OPEN){reject(new Error('WebSocket is not connected'));return;}
    const id=crypto.randomUUID(); pending.current.set(id,{resolve,reject}); socket.current.send(JSON.stringify({id,command,...extra}));
  });
  return { socket, frame, request };
}

function PlacementPreview({kind,request,onSpawn,onToast}:{kind:string|null;request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;onSpawn:(name:string)=>void;onToast:(message:string)=>void}) {
  const {camera,gl}=useThree(); const group=useRef<THREE.Group>(null); const mouse=useRef(new THREE.Vector2()); const position=useRef(new THREE.Vector3());
  const inside=useRef(false); const valid=useRef(false); const lastCheck=useRef(0); const lastKey=useRef(''); const candidateKey=useRef(''); const down=useRef<[number,number]|null>(null); const dragged=useRef(false); const lastPlaced=useRef(0); const colorRef=useRef<THREE.MeshStandardMaterial>(null); const [accepted,setAccepted]=useState(false);
  const raycaster=useMemo(()=>new THREE.Raycaster(),[]); const plane=useMemo(()=>new THREE.Plane(new THREE.Vector3(0,0,1),-.75),[]);
  useEffect(()=>{const canvas=gl.domElement; const move=(event:PointerEvent)=>{const rect=canvas.getBoundingClientRect();mouse.current.set(((event.clientX-rect.left)/rect.width)*2-1,-((event.clientY-rect.top)/rect.height)*2+1);inside.current=event.clientX>=rect.left&&event.clientX<=rect.right&&event.clientY>=rect.top&&event.clientY<=rect.bottom;
      if(down.current&&Math.hypot(event.clientX-down.current[0],event.clientY-down.current[1])>6)dragged.current=true;};
    const downEvent=(event:PointerEvent)=>{down.current=[event.clientX,event.clientY];dragged.current=false;};
    const place=()=>{if(!kind||performance.now()-lastPlaced.current<250||!inside.current||dragged.current||Math.abs(position.current.x)>.6||Math.abs(position.current.y)>.45)return;lastPlaced.current=performance.now();const pos=position.current.toArray();request('check_placement',{kind,pos,quat:[1,0,0,0]}).then(()=>request('spawn',{kind,pos,quat:[1,0,0,0]})).then(r=>onSpawn(String((r as {result:string}).result))).catch(e=>onToast(e.message));};
    const up=()=>{place();down.current=null;};const drop=(event:DragEvent)=>{if(!kind)return;event.preventDefault();dragged.current=false;place();};const over=(event:DragEvent)=>{if(kind)event.preventDefault();};
    window.addEventListener('pointerdown',downEvent);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);canvas.addEventListener('drop',drop);canvas.addEventListener('dragover',over);
    return()=>{window.removeEventListener('pointerdown',downEvent);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);canvas.removeEventListener('drop',drop);canvas.removeEventListener('dragover',over);};
  },[gl,kind,onSpawn,onToast,request]);
  useFrame(()=>{if(!kind||!group.current){if(group.current)group.current.visible=false;return;}group.current.visible=inside.current;
    raycaster.setFromCamera(mouse.current,camera);const hit=raycaster.ray.intersectPlane(plane,new THREE.Vector3());if(!hit){valid.current=false;return;}
    const half=kind==='box'?.035:kind==='cylinder'?.04:.035;position.current.set(hit.x,hit.y,.75+half);group.current.position.copy(position.current);
    const inTable=Math.abs(hit.x)<=.6&&Math.abs(hit.y)<=.45; if(!inTable){valid.current=false;setAccepted(false);return;}
    const key=`${kind}:${hit.x.toFixed(2)}:${hit.y.toFixed(2)}`;candidateKey.current=key;const now=performance.now();
    if(key!==lastKey.current&&now-lastCheck.current>120){lastKey.current=key;lastCheck.current=now;valid.current=false;request('check_placement',{kind,pos:position.current.toArray(),quat:[1,0,0,0]}).then(()=>{if(candidateKey.current===key){valid.current=true;setAccepted(true);}}).catch(()=>{if(candidateKey.current===key){valid.current=false;setAccepted(false);}});}
    if(colorRef.current)colorRef.current.color.set(accepted&&inTable?'#4ee19b':'#ff4f58');
  });
  const previewMaterial=<meshStandardMaterial ref={colorRef} color="#ff4f58" transparent opacity={.42}/>;
  return <group ref={group} visible={false}>{kind==='box'?<mesh><boxGeometry args={[.07,.07,.07]}/>{previewMaterial}</mesh>:kind==='cylinder'?<mesh rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.03,.03,.08,20]}/>{previewMaterial}</mesh>:<mesh><sphereGeometry args={[.035,20,14]}/>{previewMaterial}</mesh>}</group>;
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
  const [prompt,setPrompt]=useState(''); const [paused,setPaused]=useState(false); const [cameraOpen,setCameraOpen]=useState(true); const [placing,setPlacing]=useState<string|null>(null); const [toast,setToast]=useState('');
  const stableState=(s:LiveState)=>{setLive(s);setPrompt(s.prompt);}; const {frame,request}=useSocket(stableState);
  useEffect(()=>{fetch('/api/scene').then(r=>r.json()).then(setScene).catch(()=>setLive(s=>({...s,status:'API unavailable'})));},[]);
  const command=(command:string, extra:Record<string,unknown>={})=>request(command,extra).catch(e=>setToast(e.message));
  const notify=(message:string)=>{setToast(message);window.setTimeout(()=>setToast(''),2800);};
  return <main className="app"><Canvas shadows camera={{position:[1.5,-1.8,1.5],fov:45}} onCreated={({camera})=>{camera.up.set(0,0,1);camera.lookAt(0,0,.7);}}>
    <color attach="background" args={['#080c12']} /><ambientLight intensity={1.3}/><directionalLight position={[1,-2,3]} intensity={2.5} castShadow />
    {scene&&<Suspense fallback={null}><Scene data={scene} frames={frame} active={live.active_objects}/><PlacementPreview kind={placing} request={request} onSpawn={name=>{setPlacing(null);notify(`Placed ${name}`);}} onToast={notify}/></Suspense>}
    <OrbitControls makeDefault target={[0,0,.7]} />
  </Canvas>
  <header className="prompt panel"><span className={`status ${live.running?'on':''}`}/><input value={prompt} onChange={e=>{setPrompt(e.target.value);command('set_prompt',{text:e.target.value});}} placeholder="Describe what the robot should do…"/><button className="primary" onClick={()=>command('run')}>Run</button><button onClick={()=>command('stop')}>Stop</button></header>
  <aside className="toolbar panel"><button onClick={()=>command('reset')}>↺ Reset</button><button onClick={()=>{command(paused?'resume':'pause');setPaused(!paused);}}>{paused?'▶ Resume':'Ⅱ Pause'}</button></aside>
  <section className="tray panel"><div className="tray-title"><b>OBJECTS</b><span>{placing?`Click the table to place ${placing}`:'Drag an object onto the table'}</span></div><div className="cards">{(['box','cylinder','sphere'] as const).map(kind=>{const pool=scene?.pool[kind];const count=live.active_objects.filter(n=>n.startsWith(`${kind}_`)).length;return <div className={`card ${pool&&count>=pool.capacity?'disabled':''}`} key={kind} draggable={!!pool&&count<pool.capacity} onDragStart={()=>setPlacing(kind)} onDragEnd={()=>window.setTimeout(()=>setPlacing(null),0)} onClick={()=>pool&&count<pool.capacity&&setPlacing(kind)}><i className={`shape ${kind}`}/><div><b>{kind}</b><small>{count}/{pool?.capacity??10} in scene</small></div></div>;})}</div></section>
  <section className="camera panel"><button className="camera-toggle" onClick={()=>setCameraOpen(!cameraOpen)}>CAMERAS <span>{cameraOpen?'⌄':'⌃'}</span></button>{cameraOpen&&<div className="feeds"><img src="/api/camera/front.jpg"/><img src="/api/camera/top.jpg"/></div>}</section>
  <div className="connection">{live.status}</div>{toast&&<div className="toast panel">{toast}</div>}</main>;
}

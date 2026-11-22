import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Edges, OrbitControls, TransformControls, Grid } from '@react-three/drei';
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

type Geom = { id: number; name: string; body: number; body_name: string; type: number; size: number[]; pos: number[]; quat: number[]; rgba: number[]; is_pool: boolean; mesh_id?: number };
type SceneData = { geoms: Geom[]; body_order: number[]; pool: Record<string, {capacity:number; names:string[]; active:string[]}> };
type ObjectDescriptor = {name:string;kind:string;slot:string;size:number[];color:number[]};
type LiveState = { active_objects:string[]; object_slots:ObjectDescriptor[]; prompt:string; running:boolean; status:string };

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
  return { frame, request };
}

function PlacementPreview({kind,request,onSpawn,onToast}:{kind:string|null;request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;onSpawn:(name:string)=>void;onToast:(message:string)=>void}) {
  const {camera,gl}=useThree(); const group=useRef<THREE.Group>(null); const mouse=useRef(new THREE.Vector2()); const position=useRef(new THREE.Vector3());
  const inside=useRef(false); const lastCheck=useRef(0); const lastKey=useRef(''); const candidateKey=useRef(''); const down=useRef<[number,number]|null>(null); const dragged=useRef(false); const lastPlaced=useRef(0); const colorRef=useRef<THREE.MeshStandardMaterial>(null); const [accepted,setAccepted]=useState(false);
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
    raycaster.setFromCamera(mouse.current,camera);const hit=raycaster.ray.intersectPlane(plane,new THREE.Vector3());if(!hit)return;
    const half=kind==='box'?.035:kind==='cylinder'?.04:.035;position.current.set(hit.x,hit.y,.75+half);group.current.position.copy(position.current);
    const inTable=Math.abs(hit.x)<=.6&&Math.abs(hit.y)<=.45; if(!inTable){setAccepted(false);return;}
    const key=`${kind}:${hit.x.toFixed(2)}:${hit.y.toFixed(2)}`;candidateKey.current=key;const now=performance.now();
    if(key!==lastKey.current&&now-lastCheck.current>120){lastKey.current=key;lastCheck.current=now;request('check_placement',{kind,pos:position.current.toArray(),quat:[1,0,0,0]}).then(()=>{if(candidateKey.current===key)setAccepted(true);}).catch(()=>{if(candidateKey.current===key)setAccepted(false);});}
    if(colorRef.current)colorRef.current.color.set(accepted&&inTable?'#4ee19b':'#ff4f58');
  });
  const previewMaterial=<meshStandardMaterial ref={colorRef} color="#ff4f58" transparent opacity={.42}/>;
  return <group ref={group} visible={false}>{kind==='box'?<mesh><boxGeometry args={[.07,.07,.07]}/>{previewMaterial}</mesh>:kind==='cylinder'?<mesh rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.03,.03,.08,20]}/>{previewMaterial}</mesh>:<mesh><sphereGeometry args={[.035,20,14]}/>{previewMaterial}</mesh>}</group>;
}

function GeomMesh({ geom, active, selected, objectName, object, onSelect }: {geom:Geom; active:boolean;selected:boolean;objectName:string|null;object?:ObjectDescriptor;onSelect:(name:string|null)=>void}) {
  const ref = useRef<THREE.Mesh>(null);
  const rgba=object?.color??geom.rgba;const material = useMemo(() => new THREE.MeshStandardMaterial({color:new THREE.Color(rgba[0],rgba[1],rgba[2]), transparent:rgba[3]<1, opacity:rgba[3], roughness:.75}), [geom,rgba]);
  const [meshGeometry,setMeshGeometry]=useState<THREE.BufferGeometry|null>(null);
  useEffect(() => { if (ref.current) { ref.current.position.fromArray(geom.pos); ref.current.quaternion.set(geom.quat[1],geom.quat[2],geom.quat[3],geom.quat[0]); if(geom.type===3||geom.type===5)ref.current.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),Math.PI/2)); } }, [geom]);
  const size=object?.size??geom.size;
  const geometry=useMemo(()=>{
    if (geom.type===6) return new THREE.BoxGeometry(size[0]*2,size[1]*2,size[2]*2);
    if (geom.type===0) return new THREE.PlaneGeometry(size[0]>0?size[0]*2:10,size[1]>0?size[1]*2:10);
    if (geom.type===2) return new THREE.SphereGeometry(size[0],24,16);
    if (geom.type===5) return new THREE.CylinderGeometry(size[0],size[0],size[1]*2,20);
    if (geom.type===4) return new THREE.SphereGeometry(1,20,14).scale(size[0],size[1],size[2]);
    if (geom.type===3) return new THREE.CapsuleGeometry(size[0],size[1]*2,6,12);
    return new THREE.BoxGeometry(Math.max(size[0]*2,.02),Math.max(size[1]*2,.02),Math.max(size[2]*2,.02));
  },[geom,size]);
  useEffect(()=>{ if(geom.type!==7||geom.mesh_id===undefined)return; let alive=true; fetch(`/api/mesh/${geom.mesh_id}`).then(r=>r.arrayBuffer()).then(bytes=>{if(alive)setMeshGeometry(new STLLoader().parse(bytes));}); return()=>{alive=false;}; },[geom]);
  const pointerDown=useRef<[number,number]|null>(null);
  return <mesh ref={ref} geometry={geom.type===7?(meshGeometry??geometry):geometry} material={material} visible={!geom.is_pool || active} castShadow receiveShadow
    onPointerDown={objectName?e=>{pointerDown.current=[e.nativeEvent.clientX,e.nativeEvent.clientY];}:undefined}
    onPointerUp={objectName?e=>{e.stopPropagation();const down=pointerDown.current;pointerDown.current=null;if(down&&Math.hypot(e.nativeEvent.clientX-down[0],e.nativeEvent.clientY-down[1])<4)onSelect(objectName);}:undefined}>
    {selected&&<Edges scale={1.04} color="#ffe17a" threshold={12}/>}
  </mesh>;
}

function BodyGroup({bodyId, geoms, frames, activeSlots, objectNames, objectBySlot, selectedName, onSelect}:{bodyId:number;geoms:Geom[];frames:React.MutableRefObject<Float32Array|null>;activeSlots:Set<string>;objectNames:Map<string,string>;objectBySlot:Map<string,ObjectDescriptor>;selectedName:string|null;onSelect:(name:string|null)=>void}) {
  const bodyRef=useRef<THREE.Group>(null);
  useFrame(()=>{const pose=frames.current; if(!pose||!bodyRef.current)return; const offset=bodyId*7; bodyRef.current.position.set(pose[offset],pose[offset+1],pose[offset+2]); bodyRef.current.quaternion.set(pose[offset+4],pose[offset+5],pose[offset+6],pose[offset+3]);});
  return <group ref={bodyRef}>{geoms.map(g=>{const slot=g.name.replace('_geom','');const objectName=objectNames.get(slot)??null;return <GeomMesh key={g.id} geom={g} active={!g.is_pool||activeSlots.has(slot)} object={objectBySlot.get(slot)} objectName={objectName} selected={objectName===selectedName} onSelect={onSelect}/>;})}</group>;
}

function Scene({data, frames, objectSlots, selectedName, onSelect}:{data:SceneData;frames:React.MutableRefObject<Float32Array|null>;objectSlots:ObjectDescriptor[];selectedName:string|null;onSelect:(name:string|null)=>void}) {
  const activeSlots = useMemo(() => new Set(Object.values(data.pool).flatMap(p=>p.names.filter((_,i)=>p.active.includes(p.names[i])))),[data]);
  const slotsFromState=useMemo(()=>new Set(objectSlots.map(item=>item.slot)),[objectSlots]);
  const visibleSlots=useMemo(()=>new Set([...activeSlots,...slotsFromState]),[activeSlots,slotsFromState]);
  const objectNames=useMemo(()=>new Map(objectSlots.map(item=>[item.slot,item.name])),[objectSlots]);
  const objectBySlot=useMemo(()=>new Map(objectSlots.map(item=>[item.slot,item])),[objectSlots]);
  const byBody=useMemo(()=>data.geoms.reduce<Record<number,Geom[]>>((groups,g)=>{(groups[g.body]??=[]).push(g);return groups;},{}),[data]);
  return <>
    {Object.entries(byBody).map(([id,geoms])=><BodyGroup key={id} bodyId={Number(id)} geoms={geoms} frames={frames} activeSlots={visibleSlots} objectNames={objectNames} objectBySlot={objectBySlot} selectedName={selectedName} onSelect={onSelect}/>)}
    <Grid position={[0,0,.751]} args={[2,2]} cellSize={.1} cellThickness={.5} sectionSize={.5} sectionThickness={1} rotation={[Math.PI/2,0,0]} />
  </>;
}

function InViewObjectsMonitor({objects,scene,frames,onChange}:{objects:ObjectDescriptor[];scene:SceneData;frames:React.MutableRefObject<Float32Array|null>;onChange:(items:ObjectDescriptor[])=>void}) {
  const {camera}=useThree();
  const frustum=useMemo(()=>new THREE.Frustum(),[]);
  const viewProjection=useMemo(()=>new THREE.Matrix4(),[]);
  const center=useMemo(()=>new THREE.Vector3(),[]);
  const bounds=useMemo(()=>new THREE.Sphere(),[]);
  const lastUpdate=useRef(0);
  const lastNames=useRef('');
  const geomBySlot=useMemo(()=>new Map(scene.geoms.filter(g=>g.is_pool).map(g=>[g.name.replace('_geom',''),g])),[scene]);
  useFrame(()=>{
    const now=performance.now();
    if(now-lastUpdate.current<160||!frames.current)return;
    lastUpdate.current=now;
    camera.updateMatrixWorld();
    viewProjection.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(viewProjection);
    const visible=objects.filter(item=>{
      const geom=geomBySlot.get(item.slot);
      if(!geom)return false;
      const offset=geom.body*7;
      center.set(frames.current![offset],frames.current![offset+1],frames.current![offset+2]);
      const radius=item.kind==='box'?Math.hypot(...item.size):item.kind==='cylinder'?Math.hypot(item.size[0],item.size[1]):item.size[0];
      bounds.center.copy(center);
      bounds.radius=radius;
      return frustum.intersectsSphere(bounds);
    });
    const names=visible.map(item=>item.name).join('|');
    if(names!==lastNames.current){lastNames.current=names;onChange(visible);}
  });
  return null;
}

function InViewObjectsList({objects,selectedName,onSelect}:{objects:ObjectDescriptor[];selectedName:string|null;onSelect:(name:string)=>void}) {
  return <section className="scene-objects panel">
    <div className="scene-objects-title"><b>IN VIEW</b><span>{objects.length}</span></div>
    {objects.length===0?<p>No placed objects in view</p>:<div className="scene-object-rows">{objects.map(item=><button className={`scene-object-row ${selectedName===item.name?'selected':''}`} key={item.name} onClick={()=>onSelect(item.name)}><i className={`shape ${item.kind}`}/><span>{item.name}</span><small>{item.kind}</small></button>)}</div>}
  </section>;
}

function objectPose(item:ObjectDescriptor, scene:SceneData, frames:Float32Array|null) {
  const geom=scene.geoms.find(g=>g.name===`${item.slot}_geom`);const offset=(geom?.body??0)*7;
  if(!frames||frames.length<offset+7)return {pos:[0,0,0],quat:[1,0,0,0]};
  return {pos:[frames[offset],frames[offset+1],frames[offset+2]],quat:[frames[offset+3],frames[offset+4],frames[offset+5],frames[offset+6]]};
}

function GhostGizmo({item,scene,frames,snap,mode,request,onToast}:{item:ObjectDescriptor;scene:SceneData;frames:React.MutableRefObject<Float32Array|null>;snap:boolean;mode:'translate'|'rotate';request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;onToast:(message:string)=>void}) {
  const ghost=useRef<THREE.Group>(null!);const dragging=useRef(false);const changed=useRef(false);
  const half= item.kind==='box'?item.size[2]:item.kind==='cylinder'?item.size[1]:item.size[0];
  useFrame(()=>{if(!ghost.current)return;const pose=objectPose(item,scene,frames.current);if(!dragging.current){ghost.current.position.fromArray(pose.pos);ghost.current.quaternion.set(pose.quat[1],pose.quat[2],pose.quat[3],pose.quat[0]);}else if(snap&&mode==='translate')ghost.current.position.z=.75+half;});
  const finish=()=>{dragging.current=false;if(!ghost.current||!changed.current)return;changed.current=false;const q=ghost.current.quaternion;const quat=[q.w,q.x,q.y,q.z];const pos=ghost.current.position.toArray();request('move',{name:item.name,pos,quat}).catch(e=>onToast(e.message));};
  const size=item.size;
  return <>
    <group ref={ghost}>
      {item.kind==='box'?<mesh><boxGeometry args={[2*size[0],2*size[1],2*size[2]]}/><meshStandardMaterial color="#ffe17a" transparent opacity={.24} depthWrite={false}/><Edges scale={1.06} color="#ffe17a"/></mesh>:item.kind==='cylinder'?<mesh rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[size[0],size[0],2*size[1],24]}/><meshStandardMaterial color="#ffe17a" transparent opacity={.28} depthWrite={false}/><Edges scale={1.06} color="#ffe17a"/></mesh>:<mesh><sphereGeometry args={[size[0],24,16]}/><meshStandardMaterial color="#ffe17a" transparent opacity={.28} depthWrite={false}/><Edges scale={1.06} color="#ffe17a"/></mesh>}
    </group>
    <TransformControls object={ghost} mode={mode} showX={mode==='translate'} showY={mode==='translate'} showZ={true} onMouseDown={()=>{dragging.current=true;changed.current=false;}} onObjectChange={()=>{changed.current=true;}} onMouseUp={finish}/>
  </>;
}

function Inspector({item,scene,frames,request,onToast,onDelete,snap,setSnap,mode,setMode}:{item:ObjectDescriptor;scene:SceneData;frames:React.MutableRefObject<Float32Array|null>;request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;onToast:(message:string)=>void;onDelete:()=>void;snap:boolean;setSnap:(value:boolean)=>void;mode:'translate'|'rotate';setMode:(value:'translate'|'rotate')=>void}) {
  const [position,setPosition]=useState([0,0,0]);const [yaw,setYaw]=useState(0);const [color,setColor]=useState('#ffffff');const [size,setSize]=useState(item.size);
  useEffect(()=>{const pose=objectPose(item,scene,frames.current);const [w,x,y,z]=pose.quat;setPosition(pose.pos);setYaw(Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))*180/Math.PI);setColor(`#${item.color.slice(0,3).map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('')}`);setSize(item.size);},[item.name,scene,frames]);
  const updatePosition=(index:number,value:number)=>setPosition(p=>p.map((v,i)=>i===index?value:v));
  const sendMove=()=>{const radians=yaw*Math.PI/180;request('move',{name:item.name,pos:position,quat:[Math.cos(radians/2),0,0,Math.sin(radians/2)]}).catch(e=>onToast(e.message));};
  const sendColor=(value:string)=>{setColor(value);const hex=value.slice(1);const rgba=[0,2,4].map(i=>parseInt(hex.slice(i,i+2),16)/255);request('update_object',{name:item.name,color:[...rgba,1]}).catch(e=>onToast(e.message));};
  const sendSize=(index:number,value:number)=>{const next=size.map((v,i)=>i===index?value:v);setSize(next);request('update_object',{name:item.name,size:next}).catch(e=>onToast(e.message));};
  const labels=item.kind==='box'?['X half-size','Y half-size','Z half-size']:item.kind==='cylinder'?['Radius','Half-height']:['Radius'];
  return <aside className="inspector panel"><div className="inspector-head"><div><small>OBJECT INSPECTOR</small><h2>{item.name}</h2></div><button onClick={onDelete}>Delete</button></div><div className="kind-pill">{item.kind}</div>
    <div className="gizmo-options"><button className={mode==='translate'?'chosen':''} onClick={()=>setMode('translate')}>Move</button><button className={mode==='rotate'?'chosen':''} onClick={()=>setMode('rotate')}>Yaw</button><label><input type="checkbox" checked={snap} onChange={e=>setSnap(e.target.checked)}/> Snap to table</label></div>
    <label>Position</label><div className="number-grid">{['X','Y','Z'].map((axis,index)=><input key={axis} aria-label={`Position ${axis}`} type="number" step="0.01" value={position[index].toFixed(3)} onChange={e=>updatePosition(index,Number(e.target.value))}/>)}</div>
    <label>Yaw · degrees</label><div className="yaw-row"><input type="number" step="1" value={yaw.toFixed(1)} onChange={e=>setYaw(Number(e.target.value))}/><button onClick={sendMove}>Move</button></div>
    <label>Color</label><input className="color-input" type="color" value={color} onChange={e=>sendColor(e.target.value)}/>
    {labels.map((label,index)=><label className="range-label" key={label}>{label}<input type="range" min="0.01" max={item.kind==='sphere'?.1:.12} step="0.005" value={size[index]} onChange={e=>sendSize(index,Number(e.target.value))}/><span>{size[index].toFixed(3)} m</span></label>)}
  </aside>;
}

export function App() {
  const [scene,setScene]=useState<SceneData|null>(null); const [live,setLive]=useState<LiveState>({active_objects:[],object_slots:[],prompt:'',running:false,status:'connecting'});
  const [prompt,setPrompt]=useState(''); const [paused,setPaused]=useState(false); const [cameraOpen,setCameraOpen]=useState(true); const [placing,setPlacing]=useState<string|null>(null); const [toast,setToast]=useState('');const [selectedName,setSelectedName]=useState<string|null>(null);const [inViewObjects,setInViewObjects]=useState<ObjectDescriptor[]>([]);const [snap,setSnap]=useState(true);const [gizmoMode,setGizmoMode]=useState<'translate'|'rotate'>('translate');
  const stableState=(s:LiveState)=>{setLive(s);setPrompt(s.prompt);}; const {frame,request}=useSocket(stableState);
  useEffect(()=>{fetch('/api/scene').then(r=>r.json()).then(setScene).catch(()=>setLive(s=>({...s,status:'API unavailable'})));},[]);
  const command=(command:string, extra:Record<string,unknown>={})=>request(command,extra).catch(e=>setToast(e.message));
  const notify=(message:string)=>{setToast(message);window.setTimeout(()=>setToast(''),2800);};
  useEffect(()=>{const key=(event:KeyboardEvent)=>{if(event.key==='Escape'){setSelectedName(null);setPlacing(null);return;}if((event.key==='Delete'||event.key==='Backspace')&&selectedName){const target=event.target;if(target instanceof HTMLInputElement||target instanceof HTMLTextAreaElement)return;event.preventDefault();request('delete',{name:selectedName}).then(()=>setSelectedName(null)).catch(e=>notify(e.message));}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[selectedName,request]);
  const selected=live.object_slots?.find(item=>item.name===selectedName)??null;
  const deleteSelected=()=>{if(selectedName)request('delete',{name:selectedName}).then(()=>setSelectedName(null)).catch(e=>notify(e.message));};
  return <main className="app"><Canvas shadows camera={{position:[1.5,-1.8,1.5],fov:45}} onPointerMissed={()=>{if(!placing)setSelectedName(null);}} onCreated={({camera})=>{camera.up.set(0,0,1);camera.lookAt(0,0,.7);}}>
    <color attach="background" args={['#080c12']} /><ambientLight intensity={1.3}/><directionalLight position={[1,-2,3]} intensity={2.5} castShadow />
    {scene&&<Suspense fallback={null}><Scene data={scene} frames={frame} objectSlots={live.object_slots??[]} selectedName={selectedName} onSelect={setSelectedName}/><InViewObjectsMonitor objects={live.object_slots??[]} scene={scene} frames={frame} onChange={setInViewObjects}/><PlacementPreview kind={placing} request={request} onSpawn={name=>{setPlacing(null);setSelectedName(name);notify(`Placed ${name}`);}} onToast={notify}/>{selected&&<GhostGizmo item={selected} scene={scene} frames={frame} snap={snap} mode={gizmoMode} request={request} onToast={notify}/>}</Suspense>}
    <OrbitControls makeDefault target={[0,0,.7]} />
  </Canvas>
  <header className="prompt panel"><span className={`status ${live.running?'on':''}`}/><input value={prompt} onChange={e=>{setPrompt(e.target.value);command('set_prompt',{text:e.target.value});}} placeholder="Describe what the robot should do…"/><button className="primary" onClick={()=>command('run')}>Run</button><button onClick={()=>command('stop')}>Stop</button></header>
  <aside className="toolbar panel"><button onClick={()=>command('reset')}>↺ Reset</button><button onClick={()=>{command(paused?'resume':'pause');setPaused(!paused);}}>{paused?'▶ Resume':'Ⅱ Pause'}</button></aside>
  <InViewObjectsList objects={inViewObjects} selectedName={selectedName} onSelect={setSelectedName}/>
  <section className="tray panel"><div className="tray-title"><b>OBJECTS</b><span>{placing?`Click the table to place ${placing}`:'Drag an object onto the table'}</span></div><div className="cards">{(['box','cylinder','sphere'] as const).map(kind=>{const pool=scene?.pool[kind];const count=live.active_objects.filter(n=>n.startsWith(`${kind}_`)).length;return <div className={`card ${pool&&count>=pool.capacity?'disabled':''}`} key={kind} draggable={!!pool&&count<pool.capacity} onDragStart={()=>setPlacing(kind)} onDragEnd={()=>window.setTimeout(()=>setPlacing(null),0)} onClick={()=>pool&&count<pool.capacity&&setPlacing(kind)}><i className={`shape ${kind}`}/><div><b>{kind}</b><small>{count}/{pool?.capacity??10} in scene</small></div></div>;})}</div></section>
  <section className={`camera panel ${selected?'inspector-open':''}`}><button className="camera-toggle" onClick={()=>setCameraOpen(!cameraOpen)}>CAMERAS <span>{cameraOpen?'⌄':'⌃'}</span></button>{cameraOpen&&<div className="feeds"><img src="/api/camera/front.mjpg"/><img src="/api/camera/top.mjpg"/></div>}</section>
  {selected&&scene&&<Inspector item={selected} scene={scene} frames={frame} request={request} onToast={notify} onDelete={deleteSelected} snap={snap} setSnap={setSnap} mode={gizmoMode} setMode={setGizmoMode}/>}
  <div className="connection">{live.status}</div>{toast&&<div className="toast panel">{toast}</div>}</main>;
}

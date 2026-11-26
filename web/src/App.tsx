import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Edges, OrbitControls, TransformControls, Grid } from '@react-three/drei';
import { Component, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

type Geom = { id: number; name: string; body: number; body_name: string; type: number; size: number[]; pos: number[]; quat: number[]; rgba: number[]; is_pool: boolean; mesh_id?: number };
type SceneData = { geoms: Geom[]; body_order: number[]; pool: Record<string, {capacity:number; names:string[]; active:string[]}> };
type ObjectDescriptor = {name:string;kind:string;slot:string;size:number[];color:number[]};
type LiveState = { active_objects:string[]; object_slots:ObjectDescriptor[]; prompt:string; running:boolean; status:string; policy_status?:string };
type ArmTelemetry = {joint_state:number[];joint_targets:number[]};
type ArmActuator = {index:number;actuator_name:string;joint_name:string;unit:string;limited:boolean;control_range:number[]};

class CanvasErrorBoundary extends Component<{children:React.ReactNode},{error:Error|null}> {
  state: {error:Error|null} = {error:null};
  static getDerivedStateFromError(error:Error) { return {error}; }
  componentDidCatch(error:Error) { console.error('3D scene failed',error); }
  render() {
    if (this.state.error) return <div className="scene-error panel" role="alert"><b>3D scene stopped</b><span>{this.state.error.message}</span><button onClick={()=>window.location.reload()}>Reload scene</button></div>;
    return this.props.children;
  }
}

let requestSequence = 0;

function createRequestId(): string {
  requestSequence += 1;
  return globalThis.crypto?.randomUUID?.() ?? `sandtable-${Date.now()}-${requestSequence}`;
}

const meshGeometryCache = new Map<number, Promise<THREE.BufferGeometry>>();

function loadMeshGeometry(meshId:number):Promise<THREE.BufferGeometry> {
  let pending=meshGeometryCache.get(meshId);
  if(!pending){
    pending=fetch(`/api/mesh/${meshId}`).then(response=>{if(!response.ok)throw new Error(`Mesh ${meshId} request failed (${response.status})`);return response.arrayBuffer();}).then(bytes=>new STLLoader().parse(bytes));
    meshGeometryCache.set(meshId,pending);
  }
  return pending;
}

type PendingRequest = {resolve:(value:unknown)=>void;reject:(reason:Error)=>void;timer:number};

function useSocket(onState: (s: LiveState) => void, onConnection: (status:string)=>void,
                   onNotice: (notice:{code:string;names:string[]})=>void,
                   expectedBodyCount: React.MutableRefObject<number|null>,
                   onArmTelemetry: (value:ArmTelemetry)=>void) {
  const socket = useRef<WebSocket | null>(null);
  const frame = useRef<Float32Array | null>(null);
  const onStateRef = useRef(onState);
  const onConnectionRef = useRef(onConnection);
  const onNoticeRef = useRef(onNotice);
  const onArmTelemetryRef = useRef(onArmTelemetry);
  const expectedBodyCountRef = useRef(expectedBodyCount);
  const pending = useRef(new Map<string,PendingRequest>());
  onStateRef.current = onState;
  onConnectionRef.current = onConnection;
  onNoticeRef.current = onNotice;
  onArmTelemetryRef.current = onArmTelemetry;
  expectedBodyCountRef.current = expectedBodyCount;
  useEffect(() => {
    let disposed = false;
    let retryDelay = 500;
    let retryTimer = 0;
    let activeSocket: WebSocket | null = null;
    const rejectPending = (reason: Error) => {
      for (const [id, request] of pending.current) {
        window.clearTimeout(request.timer);
        request.reject(reason);
        pending.current.delete(id);
      }
    };
    const connect = () => {
      if (disposed) return;
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
      activeSocket = ws;
      socket.current = ws;
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { retryDelay = 500; onConnectionRef.current('connected'); };
      ws.onmessage = (event) => {
        try {
          if (typeof event.data === 'string') {
            const msg = JSON.parse(event.data) as LiveState & {type:string;id?:string;code?:string;message?:string;names?:unknown[];joint_state?:unknown;joint_targets?:unknown};
            if (msg.type === 'ack' && msg.id) {
              const request = pending.current.get(msg.id);
              if (request) { window.clearTimeout(request.timer); request.resolve(msg); pending.current.delete(msg.id); }
            } else if (msg.type === 'error' && msg.id) {
              const request = pending.current.get(msg.id);
              if (request) { window.clearTimeout(request.timer); request.reject(new Error(`${msg.code}: ${msg.message}`)); pending.current.delete(msg.id); }
            } else if (msg.type === 'state') onStateRef.current(msg);
            else if (msg.type === 'arm' && Array.isArray(msg.joint_state) && Array.isArray(msg.joint_targets)
                     && msg.joint_state.every(Number.isFinite) && msg.joint_targets.every(Number.isFinite)) {
              onArmTelemetryRef.current({joint_state:msg.joint_state as number[],joint_targets:msg.joint_targets as number[]});
            }
            else if (msg.type === 'notice' && msg.code === 'objects_recovered' && Array.isArray(msg.names)) {
              onNoticeRef.current({code:msg.code,names:msg.names.filter((name):name is string=>typeof name==='string')});
            }
            return;
          }
          if (!(event.data instanceof ArrayBuffer) || event.data.byteLength % 4 !== 0) {
            throw new Error('Invalid pose frame encoding');
          }
          const nextFrame = new Float32Array(event.data);
          const bodyCount = expectedBodyCountRef.current.current;
          if ((bodyCount !== null && nextFrame.length !== bodyCount * 7) || !nextFrame.every(Number.isFinite)) {
            throw new Error('Invalid pose frame dimensions or values');
          }
          frame.current = nextFrame;
        } catch (error) {
          onConnectionRef.current(error instanceof Error ? `protocol error: ${error.message}` : 'protocol error');
          ws.close();
        }
      };
      ws.onclose = () => {
        if (activeSocket !== ws) return;
        socket.current = null;
        rejectPending(new Error('WebSocket disconnected'));
        if (!disposed) {
          onConnectionRef.current('reconnecting');
          retryTimer = window.setTimeout(connect, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 5000);
        }
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(retryTimer);
      activeSocket?.close();
      if (socket.current === activeSocket) socket.current = null;
      rejectPending(new Error('WebSocket closed'));
    };
  }, []);
  const request=useCallback((command:string, extra:Record<string,unknown>={})=>new Promise<unknown>((resolve,reject)=>{
    if(socket.current?.readyState!==WebSocket.OPEN){reject(new Error('WebSocket is not connected'));return;}
    const id=createRequestId();
    const timer=window.setTimeout(()=>{pending.current.delete(id);reject(new Error(`Timed out waiting for ${command}`));},5000);
    pending.current.set(id,{resolve,reject,timer});
    try { socket.current.send(JSON.stringify({id,command,...extra})); }
    catch(error) { window.clearTimeout(timer);pending.current.delete(id);reject(error instanceof Error?error:new Error(String(error))); }
  }),[]);
  return { frame, request };
}

function PlacementPreview({kind,request,onSpawn,onToast}:{kind:string|null;request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;onSpawn:(name:string)=>void;onToast:(message:string)=>void}) {
  const {camera,gl}=useThree(); const group=useRef<THREE.Group>(null); const mouse=useRef(new THREE.Vector2()); const position=useRef(new THREE.Vector3());
  const inside=useRef(false); const lastCheck=useRef(0); const lastKey=useRef(''); const candidateKey=useRef(''); const down=useRef<[number,number]|null>(null); const dragged=useRef(false); const lastPlaced=useRef(0); const colorRef=useRef<THREE.MeshStandardMaterial>(null); const accepted=useRef(false);
  const raycaster=useMemo(()=>new THREE.Raycaster(),[]); const plane=useMemo(()=>new THREE.Plane(new THREE.Vector3(0,0,1),-.75),[]); const hitPoint=useMemo(()=>new THREE.Vector3(),[]);
  useEffect(()=>{accepted.current=false;lastKey.current='';candidateKey.current='';lastCheck.current=0;},[kind]);
  useEffect(()=>{const canvas=gl.domElement; const move=(event:PointerEvent)=>{const rect=canvas.getBoundingClientRect();mouse.current.set(((event.clientX-rect.left)/rect.width)*2-1,-((event.clientY-rect.top)/rect.height)*2+1);inside.current=event.clientX>=rect.left&&event.clientX<=rect.right&&event.clientY>=rect.top&&event.clientY<=rect.bottom;
      if(down.current&&Math.hypot(event.clientX-down.current[0],event.clientY-down.current[1])>6)dragged.current=true;};
    const downEvent=(event:PointerEvent)=>{down.current=[event.clientX,event.clientY];dragged.current=false;};
    const place=()=>{if(!kind||performance.now()-lastPlaced.current<250||!inside.current||dragged.current||Math.abs(position.current.x)>.6||Math.abs(position.current.y)>.45)return;lastPlaced.current=performance.now();const pos=position.current.toArray();request('check_placement',{kind,pos,quat:[1,0,0,0]}).then(()=>request('spawn',{kind,pos,quat:[1,0,0,0]})).then(r=>onSpawn(String((r as {result:string}).result))).catch(e=>onToast(e.message));};
    const up=()=>{place();down.current=null;};const drop=(event:DragEvent)=>{if(!kind)return;event.preventDefault();dragged.current=false;place();};const over=(event:DragEvent)=>{if(kind)event.preventDefault();};
    window.addEventListener('pointerdown',downEvent);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);canvas.addEventListener('drop',drop);canvas.addEventListener('dragover',over);
    return()=>{window.removeEventListener('pointerdown',downEvent);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);canvas.removeEventListener('drop',drop);canvas.removeEventListener('dragover',over);};
  },[gl,kind,onSpawn,onToast,request]);
  useFrame(()=>{if(!kind||!group.current){if(group.current)group.current.visible=false;return;}group.current.visible=inside.current;
    raycaster.setFromCamera(mouse.current,camera);const hit=raycaster.ray.intersectPlane(plane,hitPoint);if(!hit)return;
    const half=kind==='box'?.035:kind==='cylinder'?.04:.035;position.current.set(hit.x,hit.y,.75+half);group.current.position.copy(position.current);
    const inTable=Math.abs(hit.x)<=.6&&Math.abs(hit.y)<=.45; if(!inTable){accepted.current=false;lastKey.current='';candidateKey.current='';if(colorRef.current)colorRef.current.color.set('#ff4f58');return;}
    const key=`${kind}:${hit.x.toFixed(2)}:${hit.y.toFixed(2)}`;candidateKey.current=key;const now=performance.now();
    if(key!==lastKey.current&&now-lastCheck.current>120){lastKey.current=key;lastCheck.current=now;request('check_placement',{kind,pos:position.current.toArray(),quat:[1,0,0,0]}).then(()=>{if(candidateKey.current===key)accepted.current=true;}).catch(()=>{if(candidateKey.current===key)accepted.current=false;});}
    if(colorRef.current)colorRef.current.color.set(accepted.current?'#4ee19b':'#ff4f58');
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
  useEffect(()=>{ if(geom.type!==7||geom.mesh_id===undefined)return; let alive=true; loadMeshGeometry(geom.mesh_id).then(value=>{if(alive)setMeshGeometry(value);}).catch(error=>console.error('Mesh load failed',error)); return()=>{alive=false;}; },[geom]);
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

function ArmControlPanel({request,telemetryRef,onToast}:{request:(c:string,e?:Record<string,unknown>)=>Promise<unknown>;telemetryRef:React.MutableRefObject<ArmTelemetry|null>;onToast:(message:string)=>void}) {
  const [actuators,setActuators]=useState<ArmActuator[]>([]);
  const [targets,setTargets]=useState<number[]>([]);
  const [actual,setActual]=useState<number[]>([]);
  const targetRef=useRef<number[]>([]); const dragging=useRef(false); const sendTimer=useRef<number|null>(null);
  const adoptTargets=(values:number[])=>{targetRef.current=values;setTargets(values);};
  useEffect(()=>{let alive=true;fetch('/api/arm').then(response=>{if(!response.ok)throw new Error(`Arm controls request failed (${response.status})`);return response.json() as Promise<{actuators:ArmActuator[];joint_state:number[];joint_targets:number[]}>;}).then(data=>{if(!alive)return;setActuators(data.actuators);setActual(data.joint_state);adoptTargets(data.joint_targets);}).catch(error=>{if(alive)onToast(error instanceof Error?error.message:String(error));});return()=>{alive=false;if(sendTimer.current!==null)window.clearTimeout(sendTimer.current);};},[onToast]);
  useEffect(()=>{const timer=window.setInterval(()=>{const sample=telemetryRef.current;if(!sample)return;setActual(sample.joint_state);if(!dragging.current&&sendTimer.current===null)adoptTargets(sample.joint_targets);},100);return()=>window.clearInterval(timer);},[telemetryRef]);
  const send=(values:number[])=>{void request('set_arm',{targets:values}).catch(error=>{onToast(error instanceof Error?error.message:String(error));const sample=telemetryRef.current;if(sample)adoptTargets(sample.joint_targets);dragging.current=false;});};
  const change=(index:number,value:number)=>{const next=targetRef.current.slice();next[index]=value;targetRef.current=next;setTargets(next);dragging.current=true;if(sendTimer.current!==null)window.clearTimeout(sendTimer.current);sendTimer.current=window.setTimeout(()=>{sendTimer.current=null;send(next);},70);};
  const commit=()=>{if(sendTimer.current!==null){window.clearTimeout(sendTimer.current);sendTimer.current=null;}dragging.current=false;send(targetRef.current);};
  const hold=()=>request('hold_arm').catch(error=>onToast(error instanceof Error?error.message:String(error)));
  return <aside className="arm-controls panel"><div className="arm-controls-head"><div><small>SIMULATED ARM</small><b>Joint controls</b></div><button onClick={()=>void hold()}>Hold current</button></div>
    {actuators.length===0?<p>Loading arm controls…</p>:actuators.map((actuator,index)=>{const limits=actuator.control_range;const value=targets[index]??actual[index]??0;return <label className="arm-joint" key={actuator.actuator_name}><span>{actuator.joint_name.replaceAll('_',' ')}</span><b>{(value*180/Math.PI).toFixed(0)}°</b><input aria-label={actuator.joint_name.replaceAll('_',' ')} type="range" min={limits[0]} max={limits[1]} step="0.01" value={value} onChange={event=>change(index,Number(event.target.value))} onPointerUp={commit} onKeyUp={commit} onBlur={()=>{if(dragging.current)commit();}}/><small>Actual {(actual[index]??0)*180/Math.PI|0}° · limit {limits[0]*180/Math.PI|0}° to {limits[1]*180/Math.PI|0}°</small></label>;})}
    <p className="arm-control-note">Position targets · values shown in degrees</p>
  </aside>;
}

export function App() {
  const [scene,setScene]=useState<SceneData|null>(null); const [live,setLive]=useState<LiveState>({active_objects:[],object_slots:[],prompt:'',running:false,status:'connecting',policy_status:'offline'});
  const [prompt,setPrompt]=useState(''); const [paused,setPaused]=useState(false); const [cameraOpen,setCameraOpen]=useState(true); const [placing,setPlacing]=useState<string|null>(null); const [toast,setToast]=useState('');const [selectedName,setSelectedName]=useState<string|null>(null);const [inViewObjects,setInViewObjects]=useState<ObjectDescriptor[]>([]);const [snap,setSnap]=useState(true);const [gizmoMode,setGizmoMode]=useState<'translate'|'rotate'>('translate');const [runtimeError,setRuntimeError]=useState('');const [connectionStatus,setConnectionStatus]=useState('connecting');
  const toastTimer=useRef<number|null>(null);
  const notify=useCallback((message:string)=>{setToast(message);if(toastTimer.current!==null)window.clearTimeout(toastTimer.current);toastTimer.current=window.setTimeout(()=>{setToast('');toastTimer.current=null;},2800);},[]);
  const expectedBodyCount=useRef<number|null>(null);
  const armTelemetry=useRef<ArmTelemetry|null>(null);
  const [armOpen,setArmOpen]=useState(false);
  const sessionInput=useRef<HTMLInputElement|null>(null);
  const stableState=useCallback((s:LiveState)=>{setLive(s);setPrompt(s.prompt);setPaused(s.status==='paused');},[]);
  const stableConnection=useCallback((status:string)=>setConnectionStatus(status),[]);
  const stableNotice=useCallback((notice:{code:string;names:string[]})=>{
    if(notice.code==='objects_recovered')notify(`Returned ${notice.names.join(', ')} to the tray after falling off the table`);
  },[notify]);
  const stableArmTelemetry=useCallback((value:ArmTelemetry)=>{armTelemetry.current=value;},[]);
  const {frame,request}=useSocket(stableState,stableConnection,stableNotice,expectedBodyCount,stableArmTelemetry);
  useEffect(()=>{let active=true;fetch('/api/scene').then(response=>{if(!response.ok)throw new Error(`Scene request failed (${response.status})`);return response.json() as Promise<SceneData>;}).then(data=>{if(!active)return;expectedBodyCount.current=data.body_order.length;setScene(data);}).catch(error=>{if(active)setLive(s=>({...s,status:'API unavailable'}));setRuntimeError(error instanceof Error?error.message:'Unable to load scene');});return()=>{active=false;};},[]);
  useEffect(()=>{
    const onError=(event:ErrorEvent)=>setRuntimeError(event.message||'Unexpected UI error');
    const onRejection=(event:PromiseRejectionEvent)=>setRuntimeError(String(event.reason??'Unexpected asynchronous UI error'));
    window.addEventListener('error',onError);
    window.addEventListener('unhandledrejection',onRejection);
    return()=>{window.removeEventListener('error',onError);window.removeEventListener('unhandledrejection',onRejection);};
  },[]);
  const command=useCallback((name:string, extra:Record<string,unknown>={})=>request(name,extra).catch(error=>notify(error instanceof Error?error.message:String(error))),[request,notify]);
  const onSpawn=useCallback((name:string)=>{setPlacing(null);setSelectedName(name);notify(`Placed ${name}`);},[notify]);
  useEffect(()=>()=>{if(toastTimer.current!==null)window.clearTimeout(toastTimer.current);},[]);
  useEffect(()=>{if(selectedName&&!live.object_slots.some(item=>item.name===selectedName))setSelectedName(null);},[live.object_slots,selectedName]);
  useEffect(()=>{const timer=window.setTimeout(()=>{if(prompt!==live.prompt)request('set_prompt',{text:prompt}).catch(error=>notify(error instanceof Error?error.message:String(error)));},150);return()=>window.clearTimeout(timer);},[prompt,live.prompt,request,notify]);
  useEffect(()=>{const key=(event:KeyboardEvent)=>{if(event.key==='Escape'){setSelectedName(null);setPlacing(null);return;}if((event.key==='Delete'||event.key==='Backspace')&&selectedName){const target=event.target;if(target instanceof HTMLInputElement||target instanceof HTMLTextAreaElement)return;event.preventDefault();request('delete',{name:selectedName}).then(()=>setSelectedName(null)).catch(e=>notify(e.message));}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[selectedName,request]);
  const selected=live.object_slots?.find(item=>item.name===selectedName)??null;
  const deleteSelected=()=>{if(selectedName)request('delete',{name:selectedName}).then(()=>setSelectedName(null)).catch(e=>notify(e instanceof Error?e.message:String(e)));};
  const togglePause=()=>{const next=paused?'resume':'pause';request(next).then(()=>setPaused(next==='pause')).catch(error=>notify(error instanceof Error?error.message:String(error)));};
  const runPolicy=()=>request('set_prompt',{text:prompt}).then(()=>request('run')).catch(error=>notify(error instanceof Error?error.message:String(error)));
  const saveSession=async()=>{try{const response=await fetch('/api/session');if(!response.ok)throw new Error(`Session save failed (${response.status})`);const data=await response.blob();const url=URL.createObjectURL(data);const link=document.createElement('a');link.href=url;link.download='sandtable-session.json';link.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Session saved');}catch(error){notify(error instanceof Error?error.message:String(error));}};
  const loadSession=async(file:File)=>{try{if(file.size>1024*1024)throw new Error('Session file exceeds 1 MB');const session=JSON.parse(await file.text()) as unknown;const response=await fetch('/api/session',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(session)});const result=await response.json() as {detail?:string};if(!response.ok)throw new Error(result.detail??`Session load failed (${response.status})`);setSelectedName(null);setPlacing(null);notify('Session loaded');}catch(error){notify(error instanceof Error?error.message:String(error));}finally{if(sessionInput.current)sessionInput.current.value='';}};
  return <main className="app"><CanvasErrorBoundary><Canvas shadows dpr={[1,1.5]} fallback={<div className="scene-error panel">WebGL is unavailable in this browser.</div>} camera={{position:[1.5,-1.8,1.5],fov:45}} onPointerMissed={()=>{if(!placing)setSelectedName(null);}} onCreated={({camera})=>{camera.up.set(0,0,1);camera.lookAt(0,0,.7);}}>
    <color attach="background" args={['#080c12']} /><ambientLight intensity={1.3}/><directionalLight position={[1,-2,3]} intensity={2.5} castShadow />
    {scene&&<Suspense fallback={null}><Scene data={scene} frames={frame} objectSlots={live.object_slots??[]} selectedName={selectedName} onSelect={setSelectedName}/><InViewObjectsMonitor objects={live.object_slots??[]} scene={scene} frames={frame} onChange={setInViewObjects}/><PlacementPreview kind={placing} request={request} onSpawn={onSpawn} onToast={notify}/>{selected&&<GhostGizmo item={selected} scene={scene} frames={frame} snap={snap} mode={gizmoMode} request={request} onToast={notify}/>}</Suspense>}
    <OrbitControls makeDefault target={[0,0,.7]} />
  </Canvas></CanvasErrorBoundary>
  <header className="prompt panel"><span className={`status ${live.running?'pending':''}`} title={live.policy_status??'Policy offline'}/><span className="policy-label">{(live.policy_status??'offline').toUpperCase()}</span><input value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Describe what the robot should do…"/><button className="primary" onClick={runPolicy}>Run</button><button onClick={()=>command('stop')}>Stop</button></header>
  <aside className="toolbar panel"><button onClick={()=>command('reset')}>↺ Reset</button><button onClick={togglePause}>{paused?'▶ Resume':'Ⅱ Pause'}</button><button onClick={()=>setArmOpen(open=>!open)}>{armOpen?'Close arm':'Arm controls'}</button></aside>
  <aside className="session-toolbar panel"><button onClick={saveSession}>Save</button><button onClick={()=>sessionInput.current?.click()}>Load</button><input ref={sessionInput} type="file" accept="application/json,.json" hidden onChange={event=>{const file=event.target.files?.[0];if(file)void loadSession(file);}}/></aside>
  <InViewObjectsList objects={inViewObjects} selectedName={selectedName} onSelect={setSelectedName}/>
  {armOpen&&<ArmControlPanel request={request} telemetryRef={armTelemetry} onToast={notify}/>}
  <section className="tray panel"><div className="tray-title"><b>OBJECTS</b><span>{placing?`Click the table to place ${placing}`:'Drag an object onto the table'}</span></div><div className="cards">{(['box','cylinder','sphere'] as const).map(kind=>{const pool=scene?.pool[kind];const count=live.active_objects.filter(n=>n.startsWith(`${kind}_`)).length;return <div className={`card ${pool&&count>=pool.capacity?'disabled':''}`} key={kind} draggable={!!pool&&count<pool.capacity} onDragStart={()=>setPlacing(kind)} onDragEnd={()=>window.setTimeout(()=>setPlacing(null),0)} onClick={()=>pool&&count<pool.capacity&&setPlacing(kind)}><i className={`shape ${kind}`}/><div><b>{kind}</b><small>{count}/{pool?.capacity??10} in scene</small></div></div>;})}</div></section>
  <section className={`camera panel ${selected?'inspector-open':''}`}><button className="camera-toggle" onClick={()=>setCameraOpen(!cameraOpen)}>CAMERAS <span>{cameraOpen?'⌄':'⌃'}</span></button>{cameraOpen&&<div className="feeds"><img aria-label="Front camera" src="/api/camera/front.mjpg"/><img aria-label="Top camera" src="/api/camera/top.mjpg"/><img aria-label="Wrist camera" src="/api/camera/wrist.mjpg"/></div>}</section>
  {selected&&scene&&<Inspector item={selected} scene={scene} frames={frame} request={request} onToast={notify} onDelete={deleteSelected} snap={snap} setSnap={setSnap} mode={gizmoMode} setMode={setGizmoMode}/>}
  <div className="connection">{connectionStatus} · {live.status}</div>{toast&&<div className="toast panel">{toast}</div>}{runtimeError&&<div className="runtime-error panel" role="alert"><b>The UI hit an error</b><span>{runtimeError}</span><button onClick={()=>window.location.reload()}>Reload UI</button></div>}</main>;
}

import * as THREE from "three";
import Stats from "./modules/stats.module.js";
import { CSS2DRenderer, CSS2DObject } from "./modules/CSS2DRenderer.js";

const pathPrefix = "simulator/data/";
const dataJson = "data.json";
const parentElem = document.getElementById("screens");

const shader = THREE.ShaderLib.cube;
const textures = [];
const teams = [];
const pitch = 15;
let index = 0,
  intendedHeading = 0;

const data = await (await fetch(pathPrefix + dataJson)).json();

const stats = new Stats();
initWebsocket();
animate();

function createBbox(parentElem) {
  const box = document.createElement("div");
  box.className = "bbox";
  parentElem.appendChild(box);
  return box;
}

function getDims() {
  return [760, 435];
}

function toNearest90(val) {
  return Math.round(val / 90) * 90;
}

function createCorner(x, y, z) {
  const cornerDiv = document.createElement("div");
  cornerDiv.className = "label";
  cornerDiv.textContent = "";
  const corner = new CSS2DObject(cornerDiv);
  corner.position.set(x, y, z);
  corner.center.set(0, 0);
  return corner;
}

function removeCorners(team) {
  team.convexHull.forEach((corner) => {
    team.mesh.remove(corner);
  });
  team.convexHull.length = 0;
}

// instantiate new corners
function newCorners(mesh, convexHull) {
  const corners = data[index].corners;
  corners.forEach(({ x, y, z }) => {
    const obj = createCorner(x, y, z);
    mesh.add(obj);
    convexHull.push(obj);
  });
}

function rotationDiff(h1, h2) {
  const h1norm = ((h1 % 360) + 360) % 360;
  const h2norm = ((h2 % 360) + 360) % 360;
  const diff = Math.abs(h1norm - h2norm);
  if (diff > 180) {
    return 360 - diff;
  } else {
    return diff;
  }
}

function bboxUpdate(team) {
  // force a render to ensure the bbox is updated appropriately
  team.camera.updateMatrixWorld();
  team.camera.updateProjectionMatrix();
  team.renderer.render(team.scene, team.camera);
  let max_x = 0,
    max_y = 0,
    min_x = Infinity,
    min_y = Infinity;
  const dims = getDims();
  // find smallest rect which encapsulates convex hull
  team.convexHull.forEach((corner) => {
    const pos = new THREE.Vector3().setFromMatrixPosition(corner.matrixWorld);
    pos.project(team.camera);
    pos.x = Math.round((0.5 + pos.x / 2) * (dims[0] / window.devicePixelRatio));
    pos.y = Math.round((0.5 - pos.y / 2) * (dims[1] / window.devicePixelRatio));
    max_x = max_x <= pos.x ? pos.x : max_x;
    max_y = max_y <= pos.y ? pos.y : max_y;
    min_x = min_x >= pos.x ? pos.x : min_x;
    min_y = min_y >= pos.y ? pos.y : min_y;
  });
  const left = min_x * window.devicePixelRatio;
  const top = min_y * window.devicePixelRatio;
  const width = (max_x - min_x) * window.devicePixelRatio;
  const height = (max_y - min_y) * window.devicePixelRatio;

  team.bbox.style.left = `${left}px`;
  team.bbox.style.top = `${top}px`;
  team.bbox.style.width = `${width}px`;
  team.bbox.style.height = `${height}px`;
  const invalid =
    left > dims[0] ||
    left + width <= 0 ||
    top > dims[1] ||
    top + height <= 0 ||
    rotationDiff(team.camera.rotation.y, intendedHeading) > 115;
  if (invalid) {
    team.bbox.style.display = "none";
    // console.log("invalid");
    return {
      valid: !invalid,
    };
  } else {
    team.bbox.style.display = "block";
    // adjust for the window scaling
    return {
      valid: !invalid,
      left: left * window.devicePixelRatio,
      top: top * window.devicePixelRatio,
      width: width * window.devicePixelRatio,
      height: height * window.devicePixelRatio,
    };
  }
}

function updateRotation(name, yaw) {
  console.log(`name: ${name}, pitch: ${pitch}, yaw: ${yaw}`);
  const team = teams.find((team) => team.teamName === name);
  team.camera.rotation.y = (-yaw * Math.PI) / 180;
  bboxUpdate(team);
}

function updateIntendedHeading() {
  intendedHeading = toNearest90(parseInt(data[index].truth.heading));
  console.log(intendedHeading);
}

function init(names) {
  // FPS tracker
  const container = document.getElementById("container");
  container.appendChild(stats.dom);

  // Textures
  const loader = new THREE.CubeTextureLoader();
  loader.setPath(pathPrefix + "cubemaps/");

  data.forEach((testCase) => {
    textures.push(loader.load(testCase.files));
  });
  shader.uniforms.tCube.value = textures[index];
  updateIntendedHeading();
  const material = new THREE.ShaderMaterial({
    fragmentShader: shader.fragmentShader,
    vertexShader: shader.vertexShader,
    uniforms: shader.uniforms,
    depthWrite: false,
    side: THREE.BackSide,
  });

  // create for each team
  names.forEach((teamName) => {
    // create camera
    const camera = new THREE.PerspectiveCamera(
      70,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 0);
    camera.rotation.order = "YXZ";
    camera.rotation.x = (pitch * Math.PI) / 180;

    // create scene + environment box
    const scene = new THREE.Scene();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(100, 100, 100), material);
    scene.add(mesh);

    // create corners of target object
    const convexHull = [];
    newCorners(mesh, convexHull);

    // create div for overall screen
    const screen = document.createElement("div");
    screen.className = "screen";
    screen.id = `screen${teamName}`;
    parentElem.appendChild(screen);

    // create sub-div for canvas
    const canvasDiv = document.createElement("div");
    canvasDiv.className = "render";
    canvasDiv.id = `canvas`;
    screen.appendChild(canvasDiv);

    // add team name
    const teamNameElem = document.createElement("p");
    teamNameElem.textContent = `Simulator View`;
    screen.appendChild(teamNameElem);

    // create bbox
    const bbox = createBbox(canvasDiv);

    // create renderer
    const renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    const dims = getDims();
    renderer.setSize(...dims);
    renderer.domElement.id = `renderer`;
    canvasDiv.appendChild(renderer.domElement);

    // create css renderer for the hidden objects which guide the bbox
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(...dims);
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.top = "0px";
    canvasDiv.appendChild(labelRenderer.domElement);

    const _team = {
      teamName,
      scene,
      camera,
      mesh,
      bbox,
      convexHull,
      renderer,
      labelRenderer,
    };
    teams.push(_team);
    bboxUpdate(_team);
  });
}

// animation rendering loop
function animate() {
  requestAnimationFrame(animate);
  render();
}

function render() {
  stats.update();
  teams.forEach((team) => {
    team.renderer.render(team.scene, team.camera);
    team.labelRenderer.render(team.scene, team.camera);
    team.camera.updateMatrixWorld();
  });
}

function initWebsocket() {
  // Create WebSocket connection.
  const socket = new WebSocket("ws://localhost:8000/ws_sim");

  // Connection opened
  socket.addEventListener("open", (event) => {
    console.log(event);
  });
  // Listen for messages
  socket.addEventListener("message", ({ data }) => {
    const event = JSON.parse(data);
    console.log("Message from server ", event);
    switch (event.type) {
      case "teams":
        init(event.teams);
        break;
      case "health":
        console.log("health");
        socket.send("ok");
        break;
      case "update":
        console.log(event);
        updateRotation(event.name, event.yaw);
        break;
      case "switch":
        console.log("switching texture");
        index += 1;
        if (index === textures.length) {
          index = 0;
        }
        updateIntendedHeading();
        shader.uniforms.tCube.value = textures[index];
        // update corners for bbox
        teams.forEach((team) => {
          removeCorners(team);
          newCorners(team.mesh, team.convexHull);
          // re-render bboxes
          bboxUpdate(team);
        });
        break;
      case "snapshot":
        const team = teams.find((team) => team.teamName === event.name);
        const _bbox = bboxUpdate(team);
        console.log(_bbox);
        socket.send(JSON.stringify(_bbox));
        team.renderer.domElement.toBlob((blob) => {
          console.log("blob");
          socket.send(blob);
        }, "image/jpeg");
        break;
    }
  });
}

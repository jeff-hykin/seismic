import { Elemental, passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import { css, components, Column, Row, askForFiles, Code, Input, Checkbox, Dropdown, popUp, cx, } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/elements.js"
import { fadeIn, fadeOut } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/animations.js"
import { showToast } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/actions.js"
import { addDynamicStyleFlags, setupStyles, createCssClass, setupClassStyles, hoverStyleHelper, combineClasses, mergeStyles, AfterSilent, removeAllChildElements } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/helpers.js"
import { zip, enumerate, count, permute, combinations, wrapAroundGet } from "https://esm.sh/gh/jeff-hykin/good-js@1.13.5.1/source/array.js"

import storageObject from "https://esm.sh/gh/jeff-hykin/storage-object@0.0.3.5/main.js"
import { makeDraggable } from "./generic_tools.js"

import { Event, trigger, everyTime, once } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/events.js"
import { pointsToFunction } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/points_to_function.js"
import { fabric, FabricCanvas } from "./fabric.js"
import { NodeCanvas, FabricNode } from "./node_canvas.js"
import { TimelineManager } from "./timeline_manager.js"
import { Button } from "./button.js"
globalThis.fabric = fabric // debugging

// goals
    // DONE: pulse animation
    // DONE: node visual connections
    // infinite pan and zoom canvas
    // cleanup the code (accessing allNodes, remove weakref)
    // a means of creating a node
    // a means of creating an edge
    // load and save connections
    // visualize the edge weights
    // a "spike generator" node
    // edges that can be made curved
        // some way to get edges to stay
    // copy and paste nodes 

    const canvasElement = NodeCanvas({ backgroundColor: "whitesmoke", jsonObjects: {
            "objects": [
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 390,
                    "top": 612,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-0",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-0": 1.0,
                        "node-1": 0.4,
                        "node-2": 0.1,
                        "node-3": -0.4,
                        "node-4": -0.4,
                        "node-5": 0.1,
                        "node-6": 0.4,
                    },
                    "inputNodeIds": [
                        "node-1"
                    ],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 221,
                    "top": 456,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-1",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-5": -0.4,
                        "node-6": 0.1,
                        "node-0": 0.4,
                        "node-1": 1.0,
                        "node-2": 0.4,
                        "node-3": 0.1,
                        "node-4": -0.4,
                    },
                    "inputNodeIds": [
                        "node-2",
                        "node-3",
                        "node-4",
                        "node-5",
                        "node-6"
                    ],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 270,
                    "top": 222,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-2",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-6": -0.4,
                        "node-0": 0.1,
                        "node-1": 0.4,
                        "node-2": 1.0,
                        "node-3": 0.4,
                        "node-4": 0.1,
                        "node-5": -0.4,
                    },
                    "inputNodeIds": [],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 473,
                    "top": 103,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-3",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-0": -0.4,
                        "node-1": 0.1,
                        "node-2": 0.4,
                        "node-3": 1.0,
                        "node-4": 0.4,
                        "node-5": 0.1,
                        "node-6": -0.4,
                    },
                    "inputNodeIds": [],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 679,
                    "top": 181,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-4",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-1": -0.4,
                        "node-2": 0.1,
                        "node-3": 0.4,
                        "node-4": 1.0,
                        "node-5": 0.4,
                        "node-6": 0.1,
                        "node-0": -0.4,
                    },
                    "inputNodeIds": [],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 756,
                    "top": 399,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-5",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-2": -0.4,
                        "node-3": 0.1,
                        "node-4": 0.4,
                        "node-5": 1.0,
                        "node-6": 0.4,
                        "node-0": 0.1,
                        "node-1": -0.4,
                    },
                    "inputNodeIds": [],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                },
                {
                    "radius": 50,
                    "startAngle": 0,
                    "endAngle": 360,
                    "counterClockwise": false,
                    "type": "FabricNode",
                    "version": "6.6.1",
                    "originX": "center",
                    "originY": "center",
                    "left": 627,
                    "top": 603,
                    "width": 100,
                    "height": 100,
                    "fill": "hsl(193.1, 100%, 50%)",
                    "stroke": "hsl(0, 0%, 82.75%)",
                    "strokeWidth": 5,
                    "strokeDashArray": null,
                    "strokeLineCap": "butt",
                    "strokeDashOffset": 0,
                    "strokeLineJoin": "miter",
                    "strokeUniform": false,
                    "strokeMiterLimit": 4,
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "flipX": false,
                    "flipY": false,
                    "opacity": 1,
                    "shadow": null,
                    "visible": true,
                    "backgroundColor": "",
                    "fillRule": "nonzero",
                    "paintFirst": "fill",
                    "globalCompositeOperation": "source-over",
                    "skewX": 0,
                    "skewY": 0,
                    "id": "node-6",
                    "label": "A",
                    "spikeThreshold": 1,
                    "energy": 0.1,
                    "energyDecayRate": 0.1,
                    "isFiring": false,
                    "outputNodeMapping": {
                        "node-3": -0.4,
                        "node-4": 0.1,
                        "node-5": 0.4,
                        "node-6": 1.0,
                        "node-0": 0.4,
                        "node-1": 0.1,
                        "node-2": -0.4,
                    },
                    "inputNodeIds": [],
                    "stableEnergyLevel": 0.1,
                    "energyAfterFiring": 0
                }
            ],
            // // FIXME: debugging
            // "objects": [
            //     // {
            //     //     "type": "rect",
            //     //     "left": 50,
            //     //     "top": 50,
            //     //     "width": 20,
            //     //     "height": 20,
            //     //     "fill": "green",
            //     //     "overlayFill": null,
            //     //     "stroke": null,
            //     //     "strokeWidth": 1,
            //     //     "strokeDashArray": null,
            //     //     "scaleX": 1,
            //     //     "scaleY": 1,
            //     //     "angle": 0,
            //     //     "flipX": false,
            //     //     "flipY": false,
            //     //     "opacity": 1,
            //     //     "selectable": true,
            //     //     "hasControls": true,
            //     //     "hasBorders": true,
            //     //     "hasRotatingPoint": false,
            //     //     "transparentCorners": true,
            //     //     "perPixelTargetFind": false,
            //     //     "rx": 0,
            //     //     "ry": 0
            //     // },
            //     // {
            //     //     "type": "circle",
            //     //     "left": 100,
            //     //     "top": 100,
            //     //     "width": 100,
            //     //     "height": 100,
            //     //     "fill": "red",
            //     //     "overlayFill": null,
            //     //     "stroke": "rgba(100,200,200)",
            //     //     "strokeWidth": 5,
            //     //     "strokeDashArray": null,
            //     //     "scaleX": 1,
            //     //     "scaleY": 1,
            //     //     "angle": 0,
            //     //     "flipX": false,
            //     //     "flipY": false,
            //     //     "opacity": 1,
            //     //     "selectable": true,
            //     //     "hasControls": true,
            //     //     "hasBorders": true,
            //     //     "hasRotatingPoint": false,
            //     //     "transparentCorners": true,
            //     //     "perPixelTargetFind": false,
            //     //     "radius": 50
            //     // },

            //     // bottom left
            //     new FabricNode({
            //         id: "node-0",
            //         label: "A", 
            //         left:202,
            //         top:809,
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             // "node-1": 1.1, // positive is excitatory, negative is inhibitory
            //         },
            //     }),

            //     // bottom of funnel
            //     new FabricNode({
            //         id: "node-1",
            //         label: "A", 
            //         left:332, 
            //         top:689, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-0": 1.1, // positive is excitatory, negative is inhibitory
            //         },
            //     }),

            //     // top of funnel
            //     new FabricNode({
            //         id: "node-2",
            //         label: "A", 
            //         left:500, 
            //         top:300, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-1": 0.5, // positive is excitatory, negative is inhibitory
            //         },
            //     }),
            //     new FabricNode({
            //         id: "node-3",
            //         label: "A", 
            //         left:700, 
            //         top:300, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-1": 0.5, // positive is excitatory, negative is inhibitory
            //         },
            //     }),
            //     new FabricNode({
            //         id: "node-4",
            //         label: "A", 
            //         left:900, 
            //         top:300, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-1": 0.5, // positive is excitatory, negative is inhibitory
            //         },
            //     }),
            //     new FabricNode({
            //         id: "node-5",
            //         label: "A", 
            //         left:900, 
            //         top:500, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-1": 0.5, // positive is excitatory, negative is inhibitory
            //         },
            //     }),
            //     new FabricNode({
            //         id: "node-6",
            //         label: "A", 
            //         left:900, 
            //         top:700, 
            //         radius:50, 
            //         spikeThreshold:1,
            //         startingEnergy:0, 
            //         energyDecayRate:0.1,
            //         outputNodeMapping: {
            //             "node-1": 0.5, // positive is excitatory, negative is inhibitory
            //         },
            //     }),
            // ],
            
            // "background": "rgba(0, 0, 0, 0)",
        }
    })
    globalThis.canvasElement = canvasElement // debugging
    


const { html } = Elemental({
    ...components,
    Node,
    FabricCanvas,
    Column, Row, askForFiles, Code, Input, Button, Checkbox, Dropdown,
})

document.body = html`
    <body font-size=15px background-color=whitesmoke overflow=none width=100vw height=100vh overflow=hidden>
        ${canvasElement}
        <Button
            style=${{position: "fixed", right: "10px", bottom: "10px"}}
            onclick=${async ()=>{
                console.log(`canvasElement is:`,canvasElement)
                if (!canvasElement.timelineManager) {
                    showToast("Sorry the canvas is not yet loaded", {close: true})
                    return
                }
                try {
                    await canvasElement.timelineManager.goForward()
                } catch (error) {
                    console.error(`error during goForward:`, error)
                }
            }}
            >
                Next
        </Button>
    </body>
`
        // <Node label="A" x=100 y=100></Node>
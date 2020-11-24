<template>
  <div
    :id="data.id"
    :class="classes"
    :style="[styles, shading]"
    :title="showErrorMessages ? messages : undefined"
    @mouseover="setShowErrorMessages(true)"
    @mouseleave="setShowErrorMessages(false)"
  >
    <div
      id="header"
      :style="titleBackground"
      @mousedown.self.stop="startDrag"
      @contextmenu.self.prevent="openContextMenu"
    >

      <span v-if="!renaming">
        {{ data.name }}
         <ArrowButton
           id="arrow-button"
           :initialUp="true"
           v-on:arrow-button-clicked="toggleShouldShowOptions"
           v-show="data.options.size > 0"
         />
      </span>
      <input
        v-else
        type="text"
        class="dark-input"
        v-model="tempName"
        placeholder="Node Name"
        v-click-outside="doneRenaming"
        @keydown.enter="doneRenaming"
      >

      <component
        :is="plugin.components.contextMenu"
        v-model="contextMenu.show"
        :x="contextMenu.x" :y="contextMenu.y"
        :items="contextMenu.items"
        @click="onContextMenu"
      ></component>
    </div>

    <div
      class="__content"
      id="content"
    >

      <!-- Outputs -->
      <div class="__outputs">
        <component
          :is="plugin.components.nodeInterface"
          v-for="(output, name) in data.outputInterfaces"
          :key="output.id"
          :name="name"
          :data="output"
        ></component>
      </div>

      <!-- Options -->
      <div v-if=shouldShowOptions class="__options">
        <template v-for="[name, option] in data.options">

          <component
            :is="plugin.components.nodeOption"
            :key="name"
            :name="name"
            :option="option"
            :componentName="option.optionComponent"
            :node="data"
            @openSidebar="openSidebar(name)"
          ></component>

          <portal :key="'sb_' + name" to="sidebar"
                  v-if="plugin.sidebar.nodeId === data.id
                        && plugin.sidebar.optionName === name
                        && option.sidebarComponent"
          >
            <component
              :is="plugin.components.nodeOption"
              :key="data.id + name"
              :name="name"
              :option="option"
              :componentName="option.sidebarComponent"
              :node="data"
            ></component>
          </portal>

        </template>
      </div>

      <!-- Inputs -->
      <div class="__inputs">
        <component
          :is="plugin.components.nodeInterface"
          v-for="(input, name) in data.inputInterfaces"
          :key="input.id"
          :name="name"
          :data="input"
        ></component>
      </div>

    </div>

  </div>
</template>

<script lang="ts">
import { Component, Watch } from 'vue-property-decorator';
import { Components } from '@baklavajs/plugin-renderer-vue';
import ArrowButton from '@/inputs/ArrowButton.vue';
import { ModelNodes } from '@/nodes/model/Types';
import IrError from '@/app/ir/checking/irError';
import { Getter } from 'vuex-class';
import { Severity } from '@/app/ir/checking/severity';
import { OverviewNodes } from '@/nodes/overview/Types';
import { CommonNodes } from '@/nodes/common/Types';
import { DataNodes } from '@/nodes/data/Types';

@Component({
  components: { ArrowButton },
})
export default class CustomNode extends Components.Node {
  @Getter('errorsMap') errorsMap!: Map<string, IrError[]>;
  private shouldShowOptions = false;

  contextMenu = {
    show: false,
    x: 0,
    y: 0,
    items: this.getContextualMenuItems(),
  };

  private currentErrors: IrError[] = [];

  get messages(): string | undefined {
    return this.currentErrors.length !== 0
      ? this.currentErrors.map((e) => e.formattedMessage).reduce((prev, curr) => `${prev}\n${curr}`)
      : undefined;
  }

  get severity(): Severity[] {
    const index = (s: Severity) => Object.keys(Severity).indexOf(s);
    return this.currentErrors.map((e) => e.severity).sort((a, b) => index(a) - index(b));
  }

  private showErrorMessages = false;

  setShowErrorMessages(show: boolean) {
    this.showErrorMessages = show;
  }

  get shading() {
    const severities = this.severity;
    if (severities.length === 0) {
      return this.selected
        ? { 'box-shadow': '0 0 0px 2px var(--blue)' }
        : { 'box-shadow': '0 0 0px 1px var(--black)' };
    }
    switch (severities[0]) {
      case Severity.Error:
        return { 'box-shadow': '0 0 1px 2px var(--red)' };
      case Severity.Warning: // WARN
        return { 'box-shadow': '0 0 1px 2px var(--yellow)' };
      default:
        return {};
    }
  }

  @Watch('errorsMap')
  private onErrorsUpdate(errorsMap: Map<string, IrError[]>) {
    this.currentErrors = errorsMap.get(this.data.id) || [];
  }

  private getContextualMenuItems() {
    const items = [{ value: 'delete', label: 'Delete' }];
    if (this.data.type !== OverviewNodes.ModelNode
      && this.data.type !== OverviewNodes.DataNode
      && this.data.type !== CommonNodes.Custom) {
      items.unshift({ value: 'rename', label: 'Rename' });
    }
    return items;
  }

  private toggleShouldShowOptions(): void {
    this.shouldShowOptions = !this.shouldShowOptions;
  }

  get titleBackground() {
    switch (this.data.type) {
      case ModelNodes.Conv1d:
      case ModelNodes.Conv2d:
      case ModelNodes.Conv3d:
      case ModelNodes.ConvTranspose1d:
      case ModelNodes.ConvTranspose2d:
      case ModelNodes.ConvTranspose3d:
      case OverviewNodes.ModelNode:
      case DataNodes.ToTensor:
      case DataNodes.Grayscale:
        return { background: 'var(--blue)' };
      case ModelNodes.MaxPool1d:
      case ModelNodes.MaxPool2d:
      case ModelNodes.MaxPool3d:
        return { background: 'var(--red)' };
      case ModelNodes.Dropout:
      case ModelNodes.Dropout2d:
      case ModelNodes.Dropout3d:
      case OverviewNodes.DataNode:
        return { background: 'var(--pink)' };
      case ModelNodes.Relu:
      case ModelNodes.Softmax:
      case ModelNodes.Softmin:
        return { background: 'var(--green)' };
      case ModelNodes.InModel:
      case ModelNodes.OutModel:
      case DataNodes.InData:
      case DataNodes.OutData:
        return { background: 'var(--purple)' };
      case ModelNodes.Linear:
      case ModelNodes.Bilinear:
        return { background: 'var(--mustard)' };
      case ModelNodes.Transformer:
        return { background: 'var(--seafoam)' };
      case CommonNodes.Custom:
        return { background: 'var(--foreground)', color: 'var(--dark-grey)' };
      default:
        return { background: 'var(--black)' };
    }
  }
}
</script>

<style scoped lang="scss">
  #arrow-button {
    position: absolute;
    top: 5px;
    right: 0;
    padding-right: 10px;
  }

  #content {
    background: var(--dark-grey);
  }

  .node {
    font-family: Roboto, serif;
    font-size: 14px;
    &:hover {
      box-shadow: 0 0 0 0.35px var(--blue);
    }
    &.--selected {
      z-index: 5;
      box-shadow: 0 0 0 1px var(--blue);
    }
    & > #header {
      color: var(--foreground);
      padding: 0.2em 0.75em;
      border-radius: 4px 4px 0 0;
      font-size: 17px;
      text-align: center;

      & > span {
        pointer-events: none;
      }
    }
  }
</style>

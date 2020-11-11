<template>
  <div
    :class="classes"
    :style="styles"
    v-show="value"
    v-click-outside="onClickOutside"
  >
    <template v-for="(item, index) in _items">

      <div v-if="item.isDivider" :key="index" class="divider"></div>

      <div
        v-else
        :key="index"
        :class="{ 'item': true, 'submenu': !!item.submenu, '--disabled': !!item.disabled }"
        @mouseenter="onMouseEnter($event, index)"
        @mouseleave="onMouseLeave($event, index)"
        @click.stop.prevent="onClick(item)"
        class="d-flex align-items-center"
      >
        <div class="flex-fill">{{ item.label }}</div>
        <div v-if="item.submenu" class="ml-3" style="line-height:1em;">&#9205;</div>
        <context-menu
          v-if="item.submenu"
          :value="activeMenu === index"
          :items="item.submenu"
          :is-nested="true"
          :is-flipped="{ x: flippedX, y: flippedY }"
          :flippable="flippable"
          @click="onChildClick"
        ></context-menu>
      </div>

    </template>
  </div>
</template>

<script lang="ts">
import { Component } from 'vue-property-decorator';
import { Components } from '@baklavajs/plugin-renderer-vue';

@Component
export default class ContextMenu extends Components.ContextMenu {
  private static ADD_NODE = 'Add Node';

  // eslint-disable-next-line no-underscore-dangle
  get _items() {
    return this.items.filter((item) => item.label !== ContextMenu.ADD_NODE)
      .map((i) => ({ ...i, hover: false }));
  }
}
</script>

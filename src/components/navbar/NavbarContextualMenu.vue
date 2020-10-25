<template>
    <div id="contextual-menu">
      <div v-for="(editor, index) in editors" :key="index">
        <VerticalMenuButton
          :label="editor.name"
          :onClick="() => switchEditor({ editorType, index})"
          :isSelected="editorType === currEditorType && index === currEditorIndex"
        />
      </div>
      <VerticalMenuButton
        :label="'+'"
        :onClick="() => newEditor({ name: 'untitled ' + editors.length, editorType})"
        :isSelected="false"
      />
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters, mapMutations } from 'vuex';
import { Editor } from '@baklavajs/core';
import EditorType from '@/EditorType';

@Component({
  components: { VerticalMenuButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
  methods: mapMutations([
    'switchEditor',
    'newEditor',
  ]),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: Editor[];
  @Prop({ required: true }) readonly editorType!: EditorType
}

</script>

<style scoped>
  #contextual-menu {
    border: 1px solid var(--grey);
  }
</style>
